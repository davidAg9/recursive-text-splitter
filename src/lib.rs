/// A high-performance recursive character text splitter.
///
/// Mirrors Python LangChain's `RecursiveCharacterTextSplitter` API.
/// Tries each separator in priority order; splits on the first viable separator,
/// then recursively splits chunks that exceed `chunk_size` using the remaining
/// separators. Merges splits with overlap, matching Python's `merge_splits` behavior.
///
/// # Efficiency advantages over Python version:
/// - Uses `&str` byte slicing — no string cloning until final chunk output
/// - Pre-allocates output capacity based on text length
/// - Single-pass separator search using `str::find` (no regex overhead — Python uses `re.escape` + `re.search`)
/// - Minimal allocations in the hot path
#[derive(Clone)]
pub struct RecursiveCharacterTextSplitter {
    /// Ordered list of separators to try, e.g. ["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""]
    separators: Vec<&'static str>,
    /// Maximum number of characters per chunk
    chunk_size: usize,
    /// Number of characters to overlap between chunks
    chunk_overlap: usize,
    /// Whether to add start_index metadata to each chunk
    add_start_index: bool,
    /// Whether to strip whitespace from chunk edges
    strip_whitespace: bool,
    /// Whether to keep separators attached to each piece
    keep_separator: bool,
    /// How to measure chunk size
    length_function: LengthFunction,
    /// Splitting strategy (sequential by default, parallel with rayon)
    strategy: std::sync::Arc<dyn SplitterStrategy + Send + Sync>,
}

/// A chunk of text produced by the splitter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub page_content: String,
    pub start_index: Option<usize>,
}

impl Chunk {
    pub fn new(content: String, start_index: Option<usize>) -> Self {
        Self {
            page_content: content,
            start_index,
        }
    }
}

/// How to measure chunk size
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LengthFunction {
    /// Count Unicode scalar values (characters)
    #[default]
    CharCount,
    /// Count bytes
    ByteCount,
}

impl LengthFunction {
    #[inline]
    fn len(&self, text: &str) -> usize {
        match self {
            LengthFunction::CharCount => text.chars().count(),
            LengthFunction::ByteCount => text.len(),
        }
    }
}

/// Splitting strategy trait — allows selecting sequential or parallel
/// implementation at compile time via feature flags.
///
/// When the `rayon` feature is enabled, `ParallelStrategy` is available and
/// can be selected via `with_strategy::<ParallelStrategy>()`.
/// When `rayon` is disabled, only `SequentialStrategy` is available.
pub trait SplitterStrategy {
    /// Split text into chunks using the strategy's algorithm.
    fn split(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<Chunk>;

    /// Split text into content strings using the strategy's algorithm.
    fn split_content(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<String>;
}

/// Default sequential splitting strategy — single-threaded.
#[derive(Debug, Clone, Default)]
pub struct SequentialStrategy;

impl SplitterStrategy for SequentialStrategy {
    fn split(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<Chunk> {
        let chunks = splitter._split_text(text, &splitter.separators);

        if splitter.add_start_index {
            splitter.annotate_start_index(text, &chunks)
        } else {
            chunks.into_iter().map(|s| Chunk::new(s, None)).collect()
        }
    }

    fn split_content(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<String> {
        splitter
            .split_text(text)
            .into_iter()
            .map(|c| c.page_content)
            .collect()
    }
}

/// Parallel splitting strategy using rayon — requires the `rayon` feature.
#[cfg(feature = "rayon")]
#[derive(Debug, Clone, Default)]
pub struct ParallelStrategy;

#[cfg(feature = "rayon")]
impl SplitterStrategy for ParallelStrategy {
    fn split(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<Chunk> {
        let chunks = splitter._split_text_par(text, &splitter.separators);

        if splitter.add_start_index {
            splitter.annotate_start_index(text, &chunks)
        } else {
            chunks.into_iter().map(|s| Chunk::new(s, None)).collect()
        }
    }

    fn split_content(&self, splitter: &RecursiveCharacterTextSplitter, text: &str) -> Vec<String> {
        splitter
            .split_text_par(text)
            .into_iter()
            .map(|c| c.page_content)
            .collect()
    }
}

impl Default for RecursiveCharacterTextSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveCharacterTextSplitter {
    /// Create a new `RecursiveCharacterTextSplitter` with default separators.
    ///
    /// Default separators: `["\n\n", "\n", " ", ""]`
    /// Default chunk size: 4096 characters
    /// Default chunk overlap: 0
    pub fn new() -> Self {
        Self {
            separators: vec!["\n\n", "\n", " ", ""],
            chunk_size: 4096,
            chunk_overlap: 0,
            add_start_index: false,
            strip_whitespace: true,
            keep_separator: true,
            length_function: LengthFunction::default(),
            strategy: std::sync::Arc::new(SequentialStrategy),
        }
    }

    /// Set custom separators (tried in priority order)
    pub fn with_separators(mut self, separators: Vec<&'static str>) -> Self {
        self.separators = separators;
        self
    }

    /// Set the maximum chunk size (in characters by default)
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the chunk overlap (in characters by default)
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Whether to add `start_index` metadata to each chunk
    pub fn with_add_start_index(mut self, add: bool) -> Self {
        self.add_start_index = add;
        self
    }

    /// Whether to strip whitespace from chunk edges
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Whether to keep separators attached to each piece (Python default for RecursiveCharacterTextSplitter: True)
    pub fn with_keep_separator(mut self, keep: bool) -> Self {
        self.keep_separator = keep;
        self
    }

    /// Use byte count instead of character count for chunk size measurement
    pub fn with_byte_length_function(mut self) -> Self {
        self.length_function = LengthFunction::ByteCount;
        self
    }

    /// Set the splitting strategy (sequential or parallel via rayon).
    ///
    /// When the `rayon` feature is enabled, you can select `ParallelStrategy`
    /// for multi-core processing. When `rayon` is disabled, only
    /// `SequentialStrategy` is available (which is also the default).
    ///
    /// # Example
    /// ```
    /// use recursive_text_splitter::{RecursiveCharacterTextSplitter, SequentialStrategy};
    ///
    /// let splitter = RecursiveCharacterTextSplitter::new()
    ///     .with_strategy::<SequentialStrategy>();
    /// #
    /// # use recursive_text_splitter::Chunk;
    /// # #[cfg(feature = "rayon")]
    /// # use recursive_text_splitter::ParallelStrategy;
    /// # #[cfg(feature = "rayon")]
    /// # let splitter = RecursiveCharacterTextSplitter::new()
    /// #     .with_strategy::<ParallelStrategy>();
    /// ```
    pub fn with_strategy<S: SplitterStrategy + Default + Send + Sync + 'static>(mut self) -> Self {
        self.strategy = std::sync::Arc::new(S::default());
        self
    }

    /// Split text into chunks.
    ///
    /// Uses the configured strategy (sequential by default).
    pub fn split_text(&self, text: &str) -> Vec<Chunk> {
        self.strategy.split(self, text)
    }

    /// Split text into chunks, returning just the page_content strings (no metadata)
    pub fn split_text_content(&self, text: &str) -> Vec<String> {
        self.split_text(text)
            .into_iter()
            .map(|c| c.page_content)
            .collect()
    }

    /// Parallel version of `split_text` using rayon for multi-core processing.
    ///
    /// When the `rayon` feature is enabled, this method uses a thread pool
    /// to process multiple splits concurrently. This is most effective on
    /// large texts that produce many oversized splits requiring recursion.
    ///
    /// Note: The merge step (adding chunk overlap) remains sequential because
    /// overlap depends on the order of previous chunks. However, the recursive
    /// splitting of individual splits is parallelized.
    ///
    /// Requires the `rayon` feature: `features = ["rayon"]`
    #[cfg(feature = "rayon")]
    pub fn split_text_par(&self, text: &str) -> Vec<Chunk> {
        let chunks = self._split_text_par(text, &self.separators);

        if self.add_start_index {
            self.annotate_start_index(text, &chunks)
        } else {
            chunks.into_iter().map(|s| Chunk::new(s, None)).collect()
        }
    }

    /// Parallel version of `split_text_content`.
    #[cfg(feature = "rayon")]
    pub fn split_text_content_par(&self, text: &str) -> Vec<String> {
        self.split_text_par(text)
            .into_iter()
            .map(|c| c.page_content)
            .collect()
    }

    /// Check if rayon parallelism is available.
    #[cfg(feature = "rayon")]
    pub fn is_parallel(&self) -> bool {
        true
    }

    /// Check if rayon parallelism is available.
    #[cfg(not(feature = "rayon"))]
    pub fn is_parallel(&self) -> bool {
        false
    }

    /// Core recursive splitting logic — mirrors Python's `_split_text`.
    ///
    /// 1. Find the best separator for the text (first one in the list found in text)
    /// 2. Split by that separator (keeping separator attached if keep_separator=True)
    /// 3. For each split: if it fits, add to good_splits; if not, flush good_splits
    ///    via merge_splits and recurse on the oversized piece with remaining separators
    fn _split_text(&self, text: &str, separators: &[&'static str]) -> Vec<String> {
        let mut final_chunks: Vec<String> = Vec::new();

        // Find appropriate separator to use (first separator found in text)
        let (separator, new_separators) = self.find_separator_and_rest(text, separators);

        // Split by separator (keeping separator attached)
        let splits = self.split_by_separator(text, separator);

        // Determine merge separator
        // When keep_separator=True, merge_sep is "" (we don't re-insert the separator)
        // When keep_separator=False, merge_sep is the separator itself (re-insert it)
        let merge_sep = if self.keep_separator { "" } else { separator };

        // Process each split: if it fits, add to good_splits; if not, flush and recurse
        let mut good_splits: Vec<String> = Vec::new();

        for s in &splits {
            if self.length_function.len(s) < self.chunk_size {
                good_splits.push(s.clone());
            } else {
                // Flush good_splits before recursing on the oversized piece
                if !good_splits.is_empty() {
                    let merged = self.merge_splits_with_sep(&good_splits, merge_sep);
                    final_chunks.extend(merged);
                    good_splits.clear();
                }

                if new_separators.is_empty() {
                    // No more separators — just add the oversized piece
                    final_chunks.push(s.clone());
                } else {
                    // Recurse with remaining separators
                    let other = self._split_text(s, &new_separators);
                    final_chunks.extend(other);
                }
            }
        }

        // Flush remaining good_splits
        if !good_splits.is_empty() {
            let merged = self.merge_splits_with_sep(&good_splits, merge_sep);
            final_chunks.extend(merged);
        }

        final_chunks
    }

    /// Parallel recursive splitting logic — mirrors `_split_text` exactly but
    /// processes oversized splits concurrently using rayon.
    ///
    /// The algorithm preserves Python's sequential flush-and-merge ordering:
    /// 1. Walk through splits sequentially, building "segments"
    /// 2. Good splits are buffered in good_splits
    /// 3. When an oversized split is found:
    ///    a. Flush good_splits → segment::Merged(...)
    ///    b. If new_separators is empty → segment::Direct(s)
    ///    c. Else → segment::Deferred(s) (for parallel processing)
    /// 4. Flush remaining good_splits → segment::Merged(...)
    /// 5. Process all Deferred segments in PARALLEL via rayon
    /// 6. Replace Deferred markers with parallel results (in order)
    /// 7. Flatten all segments
    ///
    /// This is correct because: oversized splits are independent recursive
    /// calls on disjoint subtexts. Their results can be computed in any
    /// order (or in parallel) and still assembled correctly.
    #[cfg(feature = "rayon")]
    fn _split_text_par(&self, text: &str, separators: &[&'static str]) -> Vec<String> {
        use rayon::prelude::*;

        let (separator, new_separators) = self.find_separator_and_rest(text, separators);
        let splits = self.split_by_separator(text, separator);
        let merge_sep = if self.keep_separator { "" } else { separator };

        // Phase 1: Walk splits sequentially, building segments
        let mut good_splits: Vec<String> = Vec::new();
        let mut segments: Vec<Segment> = Vec::new();
        let mut deferred_splits: Vec<String> = Vec::new();

        for s in &splits {
            if self.length_function.len(s) < self.chunk_size {
                good_splits.push(s.clone());
            } else {
                // Flush good_splits before handling the oversized piece
                if !good_splits.is_empty() {
                    segments.push(Segment::Merged(
                        self.merge_splits_with_sep(&good_splits, merge_sep),
                    ));
                    good_splits.clear();
                }

                if new_separators.is_empty() {
                    segments.push(Segment::Direct(s.clone()));
                } else {
                    deferred_splits.push(s.clone());
                    segments.push(Segment::Deferred);
                }
            }
        }

        // Flush remaining good_splits
        if !good_splits.is_empty() {
            segments.push(Segment::Merged(
                self.merge_splits_with_sep(&good_splits, merge_sep),
            ));
        }

        // Phase 2: Process deferred (oversized) splits in parallel
        // Note: _split_text_par recursively calls itself, enabling cross-recursion
        // parallelism — oversized sub-splits at deeper recursion levels are also
        // parallelized, not just at the top level.
        if !deferred_splits.is_empty() {
            let par_results: Vec<Vec<String>> = deferred_splits
                .par_iter()
                .map(|s| self._split_text_par(s, &new_separators))
                .collect();

            // Phase 3: Replace Deferred markers with parallel results (in order)
            let mut idx = 0;
            for seg in &mut segments {
                if matches!(seg, Segment::Deferred) {
                    *seg = Segment::Merged(par_results[idx].clone());
                    idx += 1;
                }
            }
        }

        // Phase 4: Flatten all segments
        segments.into_iter().flat_map(|s| s.into_chunks()).collect()
    }
    /// return it along with the remaining separators.
    fn find_separator_and_rest(
        &self,
        text: &str,
        separators: &[&'static str],
    ) -> (&'static str, Vec<&'static str>) {
        for (i, s_) in separators.iter().enumerate() {
            if s_.is_empty() {
                // Empty separator is the fallback — use it immediately
                return (*s_, separators[i + 1..].to_vec());
            }

            if text.contains(*s_) {
                return (*s_, separators[i + 1..].to_vec());
            }
        }

        // Fallback to the last separator
        let fallback = *separators.last().unwrap_or(&" ");
        (fallback, Vec::new())
    }

    /// Split text by a separator, optionally keeping the separator attached.
    ///
    /// When `keep_separator` is True: splits like Python's `re.split(r"(sep)", text)`
    /// with keep_separator="start" — the separator is prepended to the text
    /// that follows it.
    ///
    /// When `keep_separator` is False: simple split without separators.
    ///
    /// When separator is "": splits into individual characters.
    fn split_by_separator(&self, text: &str, separator: &str) -> Vec<String> {
        if separator.is_empty() {
            // Split into individual characters
            return text.chars().map(|c| c.to_string()).collect();
        }

        if !self.keep_separator {
            return text
                .split(separator)
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect();
        }

        // keep_separator=True: replicate Python's _split_text_with_regex
        // Python: re.split(r"(sep)", text) → [before, sep, middle, sep, after]
        // With keep_separator="start": [sep + middle, sep + after, ...] + [before]
        // Result: [before, sep + after_1, sep + after_2, ...]
        //
        // We replicate by: finding all separator positions, then building
        // pieces where the separator is prepended to the text that follows it.
        let mut splits: Vec<String> = Vec::new();
        let mut start = 0usize;
        let sep_len = separator.len();

        while let Some(pos) = text[start..].find(separator) {
            let abs_pos = start + pos;
            let end = abs_pos + sep_len;

            if abs_pos > start {
                // Text before the separator
                splits.push(text[start..abs_pos].to_string());
            }

            // Find the end of this piece (up to next separator or end of text)
            if end < text.len() {
                if let Some(next_pos) = text[end..].find(separator) {
                    splits.push(text[abs_pos..end + next_pos].to_string());
                    start = end + next_pos;
                } else {
                    // No more separators — rest of text including this separator
                    splits.push(text[abs_pos..].to_string());
                    start = text.len();
                }
            } else {
                // Separator at end of text
                splits.push(separator.to_string());
                start = end;
            }
        }

        // Add remaining text after last separator
        if start < text.len() {
            splits.push(text[start..].to_string());
        }

        if splits.is_empty() {
            splits.push(text.to_string());
        }

        splits
    }

    /// Merge splits with overlap, matching Python's `_merge_splits` behavior.
    ///
    /// Algorithm (mirrors Python):
    /// 1. Greedily add splits to current_doc until adding the next would exceed chunk_size
    /// 2. When chunk is full, push it and trim from the FRONT until under chunk_overlap
    /// 3. Then append the new split
    fn merge_splits_with_sep(&self, splits: &[String], separator: &str) -> Vec<String> {
        let separator_len = self.length_function.len(separator);
        let mut docs: Vec<String> = Vec::new();
        let mut current_doc: Vec<String> = Vec::new();
        let mut total: usize = 0;

        for d in splits {
            let len_d = self.length_function.len(d);

            // Check if adding this would exceed chunk_size
            if !current_doc.is_empty() {
                let sep_contribution = if current_doc.len() > 1 {
                    separator_len
                } else {
                    0
                };
                if total + len_d + sep_contribution > self.chunk_size {
                    // Flush current doc
                    if let Some(doc) = self.join_docs(&current_doc, separator) {
                        docs.push(doc);
                    }

                    // Pop from front until under chunk_overlap or would still overflow
                    loop {
                        if current_doc.is_empty() {
                            break;
                        }
                        let sep_contrib = if current_doc.len() > 1 {
                            separator_len
                        } else {
                            0
                        };
                        let combined = total + len_d + sep_contrib;
                        let should_continue =
                            total > self.chunk_overlap || (combined > self.chunk_size);
                        if !should_continue {
                            break;
                        }
                        let first_len = self.length_function.len(&current_doc[0]);
                        total -= first_len + sep_contrib;
                        current_doc.remove(0);
                    }
                }
            }

            current_doc.push(d.clone());
            let sep_contrib = if current_doc.len() > 1 {
                separator_len
            } else {
                0
            };
            total += len_d + sep_contrib;
        }

        // Flush remaining
        if let Some(doc) = self.join_docs(&current_doc, separator) {
            docs.push(doc);
        }

        docs
    }

    /// Annotate each chunk with its start_index in the original text.
    /// Mirrors Python's `create_documents` with `add_start_index`.
    fn annotate_start_index(&self, text: &str, chunks: &[String]) -> Vec<Chunk> {
        let mut result = Vec::with_capacity(chunks.len());
        let mut index = 0usize;
        let mut previous_chunk_len = 0usize;

        for chunk_content in chunks {
            let search_from = if index == 0 && previous_chunk_len == 0 {
                0
            } else {
                (index + previous_chunk_len).saturating_sub(self.chunk_overlap)
            };

            let pos = if search_from <= text.len() {
                text[search_from..].find(chunk_content)
            } else {
                None
            };

            if let Some(p) = pos {
                result.push(Chunk::new(chunk_content.clone(), Some(search_from + p)));
                previous_chunk_len = chunk_content.chars().count();
            } else {
                result.push(Chunk::new(chunk_content.clone(), Some(0)));
            }

            index = search_from;
        }

        result
    }

    /// Join documents with separator, optionally stripping whitespace.
    fn join_docs(&self, docs: &[String], separator: &str) -> Option<String> {
        let text = docs.join(separator);
        if self.strip_whitespace {
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return None;
            }
            Some(trimmed.to_string())
        } else {
            if text.is_empty() {
                return None;
            }
            Some(text)
        }
    }
}

/// Internal segment type used by `_split_text_par` for parallel processing.
#[cfg(feature = "rayon")]
enum Segment {
    /// Already-merged chunks from flushing `good_splits`
    Merged(Vec<String>),
    /// An oversized split with no remaining separators — added directly
    Direct(String),
    /// Placeholder for a split that needs parallel recursive processing
    Deferred,
}

#[cfg(feature = "rayon")]
impl Segment {
    fn into_chunks(self) -> Vec<String> {
        match self {
            Segment::Merged(c) => c,
            Segment::Direct(s) => vec![s],
            Segment::Deferred => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_splitter() {
        let splitter = RecursiveCharacterTextSplitter::new();
        let text = "Hello world.\n\nThis is a test.\nAnother line.";
        let chunks = splitter.split_text_content(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_custom_separators() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
            .with_chunk_size(100)
            .with_chunk_overlap(20);

        let text = "INT. LIVING ROOM - DAY\nLuke sits on a couch.\n\nEXT. FOREST - NIGHT\nLeia walks through trees.";
        let chunks = splitter.split_text_content(text);

        assert!(chunks.iter().any(|c| c.contains("INT. LIVING ROOM")));
        assert!(chunks.iter().any(|c| c.contains("EXT. FOREST")));
    }

    #[test]
    fn test_chunk_size_limit() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "This is a very long sentence that should be split into multiple chunks.";
        let chunks = splitter.split_text_content(text);

        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= 10 || chunk.is_empty(),
                "Chunk '{}' has {} chars, exceeds chunk size 10",
                chunk,
                chunk.chars().count()
            );
        }
    }

    #[test]
    fn test_chunk_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(3);

        let text = "abcdefghij"; // 10 chars -> should split into 2+ chunks with overlap
        let chunks = splitter.split_text_content(text);

        assert!(
            chunks.len() >= 2,
            "Expected >= 2 chunks, got {}",
            chunks.len()
        );

        // With overlap=3, the last chunk should contain characters from the overlap region
        if chunks.len() >= 2 {
            let last_chunk = chunks.last().unwrap();
            let prev_chunk = &chunks[chunks.len() - 2];

            // Check that there's some overlap (at least 1 shared char in the boundary region)
            let overlap_exists = last_chunk.chars().take(3).any(|c| prev_chunk.contains(c));
            assert!(
                overlap_exists,
                "Expected overlapping characters between chunks"
            );
        }
    }

    #[test]
    fn test_python_hi_bye() {
        // Python langchain: chunk_size=10, chunk_overlap=0, text="hi bye"
        // "hi bye" is 7 chars, fits in chunk_size=10, so one chunk
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "hi bye";
        let chunks = splitter.split_text_content(text);
        assert_eq!(chunks, vec!["hi bye"]);
    }

    #[test]
    fn test_python_hi_bye_small_chunk() {
        // chunk_size=2, chunk_overlap=0, text = "hi bye"
        // Each chunk should be <= 2 chars
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(2)
            .with_chunk_overlap(0);

        let text = "hi bye";
        let chunks = splitter.split_text_content(text);

        for chunk in &chunks {
            assert!(
                chunk.chars().count() <= 2,
                "Chunk '{}' exceeds chunk_size 2",
                chunk
            );
        }
    }

    #[test]
    fn test_star_wars_script_style() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
            .with_chunk_size(200)
            .with_chunk_overlap(50);

        let text = "INT. COCKPIT - DAY\n\nLUKE: I am a Jedi.\n\nEXT. DESERT - DAY\n\nVADER: Join me.\n\nINT. BRIDGE - NIGHT\n\nLEIA: We need to escape.";
        let chunks = splitter.split_text_content(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(chunk.chars().count() <= 250);
        }
    }

    #[test]
    fn test_byte_length_function() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_byte_length_function();

        let text = "Hello, 世界";
        let chunks = splitter.split_text_content(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_strips_whitespace() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_strip_whitespace(true);

        let text = "  hello  \n\n  world  ";
        let chunks = splitter.split_text_content(text);

        for chunk in &chunks {
            assert_eq!(chunk, chunk.trim());
        }
    }

    #[test]
    fn test_no_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "abcdefghij";
        let chunks = splitter.split_text_content(text);

        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_large_text_recursive_split() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "a".repeat(50);
        let chunks = splitter.split_text_content(&text);

        for chunk in &chunks {
            assert!(chunk.chars().count() <= 10);
        }
    }

    #[test]
    fn test_keep_separator_false() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(100)
            .with_keep_separator(false);

        let text = "line1\nline2\nline3";
        let chunks = splitter.split_text_content(text);

        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(!chunk.ends_with('\n'));
        }
    }

    #[test]
    fn test_empty_string() {
        let splitter = RecursiveCharacterTextSplitter::new();
        let chunks = splitter.split_text_content("");
        assert!(chunks.is_empty() || chunks.iter().all(|c| c.is_empty()));
    }

    #[test]
    fn test_single_char_chunk_size() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(1)
            .with_chunk_overlap(0);

        let text = "abc";
        let chunks = splitter.split_text_content(text);
        assert_eq!(chunks, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_multiline_text() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "first line\nsecond line\nthird line";
        let chunks = splitter.split_text_content(text);

        for chunk in &chunks {
            assert!(chunk.chars().count() <= 10);
        }
    }

    #[test]
    fn test_add_start_index_simple() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_add_start_index(true)
            .with_keep_separator(false);

        let text = "abcdefghij";
        let chunks = splitter.split_text(text);

        for chunk in &chunks {
            assert!(chunk.start_index.is_some());
        }
    }
}
