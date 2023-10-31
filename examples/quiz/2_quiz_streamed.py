"""A simple quiz game.

This example improves on the `1_quiz.py` example by using streaming to generate the questions. In `1_quiz.py` the
questions were generated concurrently which allowed the quiz to start quickly but meant there was a chance of duplicate
questions being generated. In this example the questions are streamed which allows us to show the first question to the
user as soon as it is ready, while still making a single query to the LLM which avoids generating duplicate questions.

The only change from `0_quiz.py` is the return type annotations of the `generate_questions` function changing from
`list[Question]` to `Iterable[Question]`. This allows us to iterate through the questions as they are generated.

---

Run this example within this directory with:

```sh
poetry run python 2_quiz_streamed.py
```

or if you have installed magentic with pip:

```sh
python 2_quiz_streamed.py
```

---

Example run:

```
Enter a topic for a quiz: NASA
Enter the number of questions: 3

1 / 3
Q: When was NASA founded?
A: 1958
Correct! The answer is: 1958

2 / 3
Q: Who was the first person to walk on the moon?
A: Neil Armstrong
Correct! The answer is: Neil Armstrong

3 / 3
Q: What is the largest planet in our solar system?
A: Jupyter
Incorrect! The correct answer is: Jupiter

Quiz complete! You scored: 66%

Congratulations on your stellar performance! You may not have reached the moon,
 but you definitely rocked that NASA quiz with a score of 66/100! Remember,
 even astronauts have their off days. Keep reaching for the stars, and who knows,
 maybe next time you'll be the one discovering a new galaxy!
 Keep up the astronomical work! 🚀🌟
```

"""

from collections.abc import Iterable

from pydantic import BaseModel

from magentic import prompt


class Question(BaseModel):
    question: str
    answer: str


@prompt("Generate {num} quiz questions about {topic}")
def generate_questions(topic: str, num: int) -> Iterable[Question]:
    ...


@prompt(
    """Return true if the user's answer is correct.
Question: {question.question}
Answer: {question.answer}
User Answer: {user_answer}"""
)
def is_answer_correct(question: Question, user_answer: str) -> bool:
    ...


@prompt(
    "Create a short and funny message of celebration or encouragment for someone who"
    " scored {score}/100 on a quiz about {topic}."
)
def create_encouragement_message(score: int, topic: str) -> str:
    ...


topic = input("Enter a topic for a quiz: ")
num_questions = int(input("Enter the number of questions: "))
questions = generate_questions(topic, num_questions)

user_points = 0
for num, question in enumerate(questions, start=1):
    print(f"\n{num} / {num_questions}")
    print(f"Q: {question.question}")
    user_answer = input("A: ")

    if is_answer_correct(question, user_answer):
        print(f"Correct! The answer is: {question.answer}")
        user_points += 1
    else:
        print(f"Incorrect! The correct answer is: {question.answer}")

score = 100 * user_points // num_questions
print(f"\nQuiz complete! You scored: {score}%\n")
print(create_encouragement_message(score, topic))
