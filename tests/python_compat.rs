use recursive_text_splitter::RecursiveCharacterTextSplitter;

/// These tests verify that the Rust implementation produces results
/// matching Python's langchain-text-splitters RecursiveCharacterTextSplitter.
/// Expected values were verified against Python 3.9 + langchain-text-splitters 0.3.11.

#[test]
fn test_python_comparison_hi_bye_cs10() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0).split_text("hi bye")
    // -> ['hi bye']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text_content("hi bye");
    assert_eq!(chunks, vec!["hi bye"]);
}

#[test]
fn test_python_comparison_hi_bye_cs2() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=2, chunk_overlap=0).split_text("hi bye")
    // -> ['hi', 'b', 'ye']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(2)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text_content("hi bye");
    assert_eq!(chunks, vec!["hi", "b", "ye"]);
}

#[test]
fn test_python_comparison_cs1_abc() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=1, chunk_overlap=0).split_text("abc")
    // -> ['a', 'b', 'c']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(1)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text_content("abc");
    assert_eq!(chunks, vec!["a", "b", "c"]);
}

#[test]
fn test_python_comparison_50a_cs10() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0).split_text("a" * 50)
    // -> ['aaaaaaaaaa', 'aaaaaaaaaa', 'aaaaaaaaaa', 'aaaaaaaaaa', 'aaaaaaaaaa']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let text = "a".repeat(50);
    let chunks = splitter.split_text_content(&text);
    let expected = vec!["aaaaaaaaaa"; 5];
    assert_eq!(chunks, expected);
}

#[test]
fn test_python_comparison_10chars_cs10() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0).split_text("abcdefghij")
    // -> ['abcdefghij']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(10)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text_content("abcdefghij");
    assert_eq!(chunks, vec!["abcdefghij"]);
}

#[test]
fn test_python_comparison_overlap() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=5, chunk_overlap=3).split_text("abcdefghij")
    // -> ['abcde', 'cdefg', 'efghi', 'ghij']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(5)
        .with_chunk_overlap(3);
    let chunks = splitter.split_text_content("abcdefghij");
    assert_eq!(chunks, vec!["abcde", "cdefg", "efghi", "ghij"]);
}

#[test]
fn test_python_comparison_empty() {
    // Python: split_text("") with any splitter -> []
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(5)
        .with_chunk_overlap(3);
    let chunks = splitter.split_text_content("");
    assert!(chunks.is_empty());
}

#[test]
fn test_python_comparison_strip_whitespace() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=5, strip_whitespace=True).split_text("  hello  \n\n  world  ")
    // -> ['hell', 'o', 'worl', 'd']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(5)
        .with_chunk_overlap(0);
    let chunks = splitter.split_text_content("  hello  \n\n  world  ");
    assert_eq!(chunks, vec!["hell", "o", "worl", "d"]);
}

#[test]
fn test_python_comparison_keep_separator_false() {
    // Python: RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, keep_separator=False)
    //         .split_text("line1\nline2\nline3")
    // -> ['line1\nline2\nline3']
    let splitter = RecursiveCharacterTextSplitter::new()
        .with_chunk_size(1000)
        .with_chunk_overlap(0)
        .with_keep_separator(false);
    let chunks = splitter.split_text_content("line1\nline2\nline3");
    assert_eq!(chunks, vec!["line1\nline2\nline3"]);
}
