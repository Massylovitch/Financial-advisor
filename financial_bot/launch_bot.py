from beam import endpoint, Image, Volume, env
import fire

if env.is_remote():
    from bot_definition import FinancialBot

def load_bot():
    
    bot = FinancialBot()
    return bot

@endpoint(
    name="financial_bot_dev",
    cpu=4,
    on_start=load_bot,
    memory="4Gi",
    secrets=["QDRANT_URL", "QDRANT_API_KEY"],
    image=Image(python_version="python3.10", python_packages="requirements.txt"),
    volumes=[
        Volume(
            mount_path="./model_cache", name="model_cache"
        ),
    ],
)
def run_dev(context):
    inputs = {
        "about_me":  "I am a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
        "history": "[[\"What is your opinion on investing in startup companies?\", \"Startup investments can be very lucrative, but they also come with a high degree of risk. It is important to do your due diligence and research the company thoroughly before investing.\"]]",
        # "context": bot,
    }

    response = _run(context, **inputs)
    return response

def _run(context, **inputs):
    
    bot = context.on_start_value
    input_payload = {
        "about_me": inputs["about_me"],
        "question": inputs["question"],
        "to_load_history": eval(inputs["history"]) if "history" in inputs else [],
    }
    response = bot.answer(**input_payload)

    return response


if __name__ == "__main__":
    fire.Fire(load_bot)