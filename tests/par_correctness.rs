use recursive_text_splitter::RecursiveCharacterTextSplitter;

#[cfg(feature = "rayon")]
mod tests {
    use super::*;

    fn generate_text(num_scenes: usize) -> String {
        let mut text = String::new();
        for i in 0..num_scenes {
            text.push_str(&format!("INT. ROOM {i} - DAY\n\n"));
            text.push_str(&format!(
                "LUKE: This is scene {i}. It has some dialogue that goes on for quite a \
                 while to make the chunk interesting. The force is strong with this one.\n\n"
            ));
            text.push_str(&format!("EXT. FOREST {i} - NIGHT\n\n"));
            text.push_str(&format!(
                "VADER: Join me. The dark side of the force is compelling and powerful \
                 beyond measure. We can rule the galaxy together.\n\n"
            ));
        }
        text
    }

    #[test]
    fn test_par_matches_seq_simple() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);

        let text = "hi bye abc def ghi";
        let seq = splitter.split_text_content(text);
        let par = splitter.split_text_content_par(text);
        assert_eq!(seq, par);
    }

    #[test]
    fn test_par_matches_seq_medium() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_separators(vec!["\nINT.", "\nEXT.", "\n\n", "\n", " ", ""])
            .with_chunk_size(200)
            .with_chunk_overlap(50);

        let text = generate_text(100);
        let seq = splitter.split_text_content(&text);
        let par = splitter.split_text_content_par(&text);
        assert_eq!(seq, par);
    }

    #[test]
    fn test_par_matches_seq_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(50)
            .with_chunk_overlap(20);

        let text = generate_text(50);
        let seq = splitter.split_text_content(&text);
        let par = splitter.split_text_content_par(&text);
        assert_eq!(seq, par);
    }
}
