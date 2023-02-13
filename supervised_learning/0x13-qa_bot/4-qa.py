#!/usr/bin/env python3
"""4-qa.py"""
semantic_search = __import__('3-semantic_search').semantic_search
question_ans = __import__('0-qa').question_answer


def question_answer(coprus_path):
    """Function that answers questions from multiple reference texts"""
    while True:
        question = input("Q: ")
        if question.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        else:
            refer = semantic_search(coprus_path, question)
            answer = question_ans(question, refer)
            if answer not in ['[CLS]', '[SEP]']:
                print("A: {}".format(answer))
            else:
                print("A: Sorry, I do not understand your question.")
