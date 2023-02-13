#!/usr/bin/env python3
"""0-qa"""
import tensorflow as tf
from transformers import BertTokenizer, TFBertForQuestionAnswering


def question_answer(question, reference):
    """
    Function that finds a snippet of text within a reference document to
    answer a question
    """
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-\
        masking-finetuned-squad')
    model = TFBertForQuestionAnswering.from_pretrained('bert-large-uncased-\
        whole-word-masking-finetuned-squad')
    input = tokenizer.encode_plus(question, reference, return_tensors='tf')
    input_ids = input['input_ids'].numpy()[0]
    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    outputs = model(input)
    answer_start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    answer_end = (tf.argmax(outputs.end_logits, axis=1) + 1).numpy()[0]
    answer = tokenizer.convert_tokens_to_string(text_tokens[answer_start:\
        answer_end])
    return answer
