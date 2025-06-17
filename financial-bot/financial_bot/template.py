import dataclasses
from typing import Dict, List, Union


@dataclasses.dataclass
class PromptTemplate:

    name: str
    system_template: str = "{system_message}"
    context_template: str = "{user_context}\n{news_context}"
    chat_history_template: str = "{chat_history}"
    question_template: str = "{question}"
    answer_template: str = "{answer}"
    system_message: str = ""
    sep: str = "\n"
    eos: str = ""

    def format_infer(self, sample):

        prompt = self.infer_raw_template.format(
            user_context=sample["user_context"],
            news_context=sample["news_context"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
        )
        return {"prompt": prompt, "payload": sample}

    @property
    def infer_raw_template(self):
        system = self.system_template.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_template}"
        chat_history = f"{self.sep}{self.chat_history_template}"
        question = f"{self.sep}{self.question_template}"

        return f"{system}{context}{chat_history}{question}{self.eos}"


templates: Dict[str, PromptTemplate] = {}


def get_llm_template(name):

    return templates[name]


def register_llm_template(template: PromptTemplate):
    """Register a new template to the global templates registry"""

    templates[template.name] = template


register_llm_template(
    PromptTemplate(
        name="falcon",
        system_template=">>INTRODUCTION<< {system_message}",
        system_message="You are a helpful assistant, with financial expertise.",
        context_template=">>DOMAIN<< {user_context}\n{news_context}",
        chat_history_template=">>SUMMARY<< {chat_history}",
        question_template=">>QUESTION<< {question}",
        answer_template=">>ANSWER<< {answer}",
        sep="\n",
        eos="<|endoftext|>",
    )
)

if __name__ == "__main__":
    print(get_llm_template("falcon"))
