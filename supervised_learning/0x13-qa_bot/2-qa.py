#!/usr/bin/env python3
""" 2. Answer Questions """
qa = __import__('0-qa').question_answer


def answer_loop(reference):
    """ Function that answers questions from a reference text """
    end = ['exit', 'quit', 'goodbye', 'bye']
    while True:
        question = input('Q: ').lower()
        if question in end:
            print('A: Goodbye')
            break
        else:
            answer = qa(question, reference)
            if answer not in ['[CLS]']:
                print('A: {}'.format(answer))
            else:
                print('A: Sorry, I do not understand your question.')
