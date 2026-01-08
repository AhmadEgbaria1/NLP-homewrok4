import sys
import os
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline


def main():
    # 1. Parse Command Line Arguments

    if len(sys.argv) < 3:
        print("Usage: python knesset_dictabert.py <path/to/masked_sampled_sents.txt> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "dictabert_results.txt")

    #print("Loading DictaBERT model... (this may take a minute)")


    # We use the Hugging Face 'pipeline' API.
    # This automatically downloads the 'dicta-il/dictabert' model and sets it up.
    # ==============================================================================
    try:
        mask_filler = pipeline("fill-mask", model="dicta-il/dictabert")
    except Exception as e:
        print(f"Error loading DictaBERT. Make sure you installed 'transformers' and 'torch'.\nDetails: {e}")
        sys.exit(1)

    print(f"Processing file: {input_path}...")

    results = []

    with open(input_path, 'r', encoding='utf-8') as f:
        # Read the file. The file format from HW2 usually has lines like:
        # "masked_sentence: ... origin: ..." or just raw sentences.


        lines = f.readlines()

    with open(output_path, 'w', encoding='utf-8') as out_f:
        for line in lines:
            original_line = line.strip()
            if not original_line:
                continue

            # 2. Preprocessing: Replace [*] with [MASK]
            masked_text = original_line.replace("[*]", "[MASK]")

            # If there are no masks (maybe a header line?), just skip or write as is
            if "[MASK]" not in masked_text:
                continue

            # 3. Prediction (The Model at Work)
            # We ask DictaBERT to fill the masks.
            # 'top_k=1' means we only want the #1 best guess for each blank.
            predictions = mask_filler(masked_text, top_k=1)

            # 4. Handle Results
            # The pipeline returns different structures depending on how many masks are in the sentence.
            # Case A: Single Mask -> returns a list of dicts
            # Case B: Multiple Masks -> returns a list of lists

            generated_tokens = []
            final_sentence = masked_text

            # Normalize to list of lists if it's a single mask, so we can use one loop
            if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], dict):
                predictions = [predictions]

                # Sort predictions by their appearance in the sentence to replace correctly


            current_token_idx = 0




            for pred_group in predictions:
                # pred_group is a list of top_k guesses for ONE specific mask.
                # Since we used top_k=1, it is a list with 1 item.
                best_guess = pred_group[0]
                token = best_guess['token_str']

                generated_tokens.append(token)

                # Replace the *first* occurrence of [MASK] with this token
                final_sentence = final_sentence.replace("[MASK]", token, 1)

            # 5. Format Output
            # masked_sentence: <original>
            # dictaBERT_sentence: <filled>
            # dictaBERT_tokens: <list>

            tokens_str = ", ".join(generated_tokens)

            out_f.write(f"masked_sentence: {original_line}\n")
            out_f.write(f"dictaBERT_sentence: {final_sentence}\n")
            out_f.write(f"dictaBERT_tokens: {tokens_str}\n")
            out_f.write("\n")  # Empty line separation

    print(f"Done! Results saved to: {output_path}")


if __name__ == "__main__":
    main()