# ragas.py
# 
# Date: May 22
#
# an implementation of the RAGAS evaluation framework described in: 
# https://arxiv.org/pdf/2309.15217
#

def faithfulness(answer: str, context: str) -> float:
    '''
    '''

    # 1. use llm to extract a set of statements from answer

    # prompt:
    # Given a question and answer, create one
    # or more statements from each sentence
    # in the given answer.
    # question: [question]
    # answer: [answer]

    # 2. For each statement in the set, use LLM to determine if the statement
    # can be inferred from the context using the prompt:

    # Consider the given context and following
    # statements, then determine whether they
    # are supported by the information present
    # in the context. Provide a brief explanation for each statement before arriving
    # at the verdict (Yes/No). Provide a final
    # verdict for each statement in order at the
    # end in the given format. Do not deviate
    # from the specified format.
    # statement: [statement 1]
    # ...
    # statement: [statement n]

    # 3. Compute faithfulness: 
    # F = the number of supported statements / the total number of statements

    ...

def answer_relevance(answer: str, question: str) -> float:
    '''
    '''

    # 