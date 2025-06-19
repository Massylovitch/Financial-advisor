# from beam import endpoint, Image, Volume, env
import fire

from financial_bot.bot_definition import FinancialBot


def load_bot():

    bot = FinancialBot()
    return bot


# @endpoint(
#     name="financial_bot_dev",
#     cpu=4,
#     on_start=load_bot,
#     memory="4Gi",
#     secrets=["QDRANT_URL", "QDRANT_API_KEY"],
#     image=Image(python_version="python3.12", python_packages="requirements.txt"),
#     volumes=[
#         Volume(mount_path="./model_cache", name="model_cache"),
#     ],
# )
def run_dev():
    bot = load_bot()
    inputs = {
        "about_me": "I am a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
        "history": [
            {
                "role": "user",
                "metadata": None,
                "content": "What's your opinion on investing in startup companies?",
                "options": None,
            },
            {
                "role": "assistant",
                "metadata": None,
                "content": "Startup investments can be very lucrative, but they also come with a high degree of risk. It is important to do your due diligence and research the company thoroughly before investing.",
                "options": None,
            },
        ],
        "context": bot,
    }

    response = _run(**inputs)
    return response


def _run(**inputs):

    bot = inputs["context"]
    input_payload = {
        "about_me": inputs["about_me"],
        "question": inputs["question"],
        "to_load_history": inputs["history"] if "history" else [],
    }
    response = bot.answer(**input_payload)

    return response


if __name__ == "__main__":
    fire.Fire(run_dev)
