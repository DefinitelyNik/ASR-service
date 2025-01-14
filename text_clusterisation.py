import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity


def combine_sentences(sentences, buffer_size=1):
    for i in range(len(sentences)):
        combined_sentence = ''

        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence'] + ' '

        combined_sentence += sentences[i]['sentence']

        for j in range(i + 1, i + 1 + buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentence'] = combined_sentence

    return sentences


def get_sentence_embedding(sentence, model, tokenizer):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def calculate_cosine_distances(sentences):
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]
        distance = 1 - similarity
        distances.append(distance)
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


def summarize(
    text, model, tokenizer, n_words=None, compression=None,
    max_length=1000, num_beams=3, do_sample=False, repetition_penalty=10.0,
    **kwargs
):
    """
    Summarize the text
    The following parameters are mutually exclusive:
    - n_words (int) is an approximate number of words to generate.
    - compression (float) is an approximate length ratio of summary and original text.
    """

    if n_words:
        text = '[{}] '.format(n_words) + text
    elif compression:
        text = '[{0:.1g}] '.format(compression) + text
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **x,
            max_length=max_length, num_beams=num_beams,
            do_sample=do_sample, repetition_penalty=repetition_penalty,
            **kwargs
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


## TODO: сделать так, чтобы у суммаризированных чанков были таймкоды
def preprocess_chunks(chunks):
    pass

def run_summarization(text):
    start_time = time.time()

    # Splitting the essay on '.', '?', and '!'
    single_sentences_list = re.split(r'(?<=[.?!])\s+', text)
    print (f"{len(single_sentences_list)} senteneces were found")
    words_count = len(text.split())
    print(f"{words_count} words were found")

    sentences = [{'sentence': x, 'index' : i} for i, x in enumerate(single_sentences_list)]
    sentences = combine_sentences(sentences)

    ## TODO: другой токенизатор
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/ruRoberta-large")
    model = AutoModel.from_pretrained("sberbank-ai/ruRoberta-large")

    embeddings = [get_sentence_embedding(sentence=x['combined_sentence'], model=model, tokenizer=tokenizer) for x in sentences]

    for i, sentence in enumerate(sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]

    distances, sentences = calculate_cosine_distances(sentences)

    # plt.plot(distances)
    #
    # y_upper_bound = .2
    # plt.ylim(0, y_upper_bound)
    # plt.xlim(0, len(distances))

    breakpoint_percentile_threshold = 85
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    # plt.axhline(y=breakpoint_distance_threshold, color='r', linestyle='-')

    # num_distances_above_theshold = len([x for x in distances if x > breakpoint_distance_threshold])
    # plt.text(x=(len(distances)*.01), y=y_upper_bound/50, s=f"{num_distances_above_theshold + 1} Chunks")

    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    # for i, breakpoint_index in enumerate(indices_above_thresh):
    #     start_index = 0 if i == 0 else indices_above_thresh[i - 1]
    #     end_index = breakpoint_index if i < len(indices_above_thresh) - 1 else len(distances)
    #
    #     plt.axvspan(start_index, end_index, facecolor=colors[i % len(colors)], alpha=0.25)
    #     plt.text(x=np.average([start_index, end_index]),
    #              y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
    #              s=f"Chunk #{i}", horizontalalignment='center',
    #              rotation='vertical')

    # if indices_above_thresh:
    #     last_breakpoint = indices_above_thresh[-1]
    #     if last_breakpoint < len(distances):
    #         plt.axvspan(last_breakpoint, len(distances), facecolor=colors[len(indices_above_thresh) % len(colors)], alpha=0.25)
    #         plt.text(x=np.average([last_breakpoint, len(distances)]),
    #                  y=breakpoint_distance_threshold + (y_upper_bound)/ 20,
    #                  s=f"Chunk #{i+1}",
    #                  rotation='vertical')

    # plt.title("PG Essay Chunks Based On Embedding Breakpoints")
    # plt.xlabel("Index of sentences in essay (Sentence Position)")
    # plt.ylabel("Cosine distance between sequential sentences")
    # plt.show()

    start_index = 0

    chunks = []

    for index in indices_above_thresh:
        end_index = index

        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([d['sentence'] for d in group])
        chunks.append(combined_text)

        start_index = index + 1

    if start_index < len(sentences):
        combined_text = ' '.join([d['sentence'] for d in sentences[start_index:]])
        chunks.append(combined_text)

    for i, chunk in enumerate(chunks):
        print (f"Chunk #{i}")
        print (chunk.strip())
        print ("\n")

    sum_model_name = 'cointegrated/rut5-base-absum'
    sum_model = T5ForConditionalGeneration.from_pretrained(sum_model_name)
    sum_tokenizer = T5Tokenizer.from_pretrained(sum_model_name)
    model.cuda()
    model.eval()

    summarized_chunkes = []

    for i, chunk in enumerate(chunks):
        print(f"Chunk #{i}")
        sum_chunk = summarize(chunk, n_words=5, model=sum_model, tokenizer=sum_tokenizer)
        summarized_chunkes.append(sum_chunk)
        print(sum_chunk)
        print("\n")

    end_time = time.time() - start_time
    print(end_time)

    return summarized_chunkes

# For testing
# with open('text.txt', 'r', encoding='utf-8') as file: # change to your file
#         text = file.read()
#
# run_summarization(text)