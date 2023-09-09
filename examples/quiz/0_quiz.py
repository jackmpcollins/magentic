"""A simple quiz game.

This example demonstrates how the `@prompt` decorator makes it easy to integrate LLMs
at multiple points in a program, for different purposes.

---

Run this example within this directory with:

```sh
poetry run python 0_quiz.py
```

or if you have installed magentic with pip:

```sh
python 0_quiz.py
```

---

Example run:

```
Enter a topic for a quiz: pizza
Enter the number of questions: 3

1 / 3
Q: What is the most popular topping on pizza?
A: cheese
Incorrect! The correct answer is: Pepperoni

2 / 3
Q: Where was pizza invented?
A: italy
Correct! The answer is: Italy

3 / 3
Q: What is the traditional shape of a pizza?
A: circular
Correct! The answer is: Round

Quiz complete! You scored: 66%

Hey pizza enthusiast! Congrats on scoring 66/100 on the pizza quiz! You may not have
 aced it, but hey, you've still got a slice of the pie! Keep up the cheesy spirit and
 remember, there's always room for improvement... and extra toppings! 🍕🎉
```

"""


from pydantic import BaseModel

from magentic import prompt


class Question(BaseModel):
    question: str
    answer: str


@prompt("Generate {num} quiz questions about {topic}")
def generate_questions(topic: str, num: int) -> list[Question]:
    ...


@prompt("""Return true if the user's answer is correct.
Question: {question.question}
Answer: {question.answer}
User Answer: {user_answer}""")
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
    print(f"\n{num} / {len(questions)}")
    print(f"Q: {question.question}")
    user_answer = input("A: ")

    if is_answer_correct(question, user_answer):
        print(f"Correct! The answer is: {question.answer}")
        user_points += 1
    else:
        print(f"Incorrect! The correct answer is: {question.answer}")

score = 100 * user_points // len(questions)
print(f"\nQuiz complete! You scored: {score}%\n")
print(create_encouragement_message(score, topic))
