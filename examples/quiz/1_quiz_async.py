"""A simple quiz game.

This example builds on the `0_quiz.py` example to demonstrate how using asyncio can greatly speed up queries. The quiz
questions are now generated concurrently which means the quiz starts much more quickly after the user has entered the
topic and number of questions. However since the questions are generated independently there is more likelihood of
duplicates - increasing the model temperature can help with this.

---

Run this example within this directory with:

```sh
poetry run python 1_quiz_async.py
```

or if you have installed magentic with pip:

```sh
python 1_quiz_async.py
```

---

Example run:

```
Enter a topic for a quiz: France
Enter the number of questions: 3

1 / 3
Q: Which French king was known as the 'Sun King'?
A: Louis XI
Incorrect! The correct answer is: Louis XIV

2 / 3
Q: What is the capital city of France?
A: Paris
Correct! The answer is: Paris

3 / 3
Q: What is the official motto of France?
A: Bonjour!
Incorrect! The correct answer is: Liberty, Equality, Fraternity

Quiz complete! You scored: 33%

Congratulations on your impressive score of 33/100 on the France quiz! Who needs to know about the Eiffel Tower or
 croissants when you can rock the world with your unique knowledge? Keep shining bright, my friend, and remember,
 there's always room for improvement...unless we're talking about French pastries, then there's never enough room!
 Keep up the great work!
```

"""


import asyncio

from pydantic import BaseModel

from magentic import prompt
from magentic.chat_model.openai_chat_model import OpenaiChatModel


class Question(BaseModel):
    question: str
    answer: str


@prompt(
    "Generate a quiz question about {topic} with difficulty {difficulty}/5.",
    # Increase temperature to try generate unique questions
    model=OpenaiChatModel(temperature=1),
)
async def generate_question(topic: str, difficulty: int) -> Question:
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


async def main() -> None:
    topic = input("Enter a topic for a quiz: ")
    num_questions = int(input("Enter the number of questions: "))
    # Generate questions concurrently
    questions = await asyncio.gather(
        *(
            generate_question(topic, difficulty=max(num + 1, 5))
            for num in range(num_questions)
        )
    )

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


asyncio.run(main())
