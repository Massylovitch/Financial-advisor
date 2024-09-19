import fire
from bot_definition import FinancialBot

def run_bot():

    bot = FinancialBot()
    
    inputs = {
        "about_me":  "I am a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
        "history": "[[\"What is your opinion on investing in startup companies?\", \"Startup investments can be very lucrative, but they also come with a high degree of risk. It is important to do your due diligence and research the company thoroughly before investing.\"]]",
        "context": bot,
    }

    response = _run(**inputs)
    return response

def _run(**inputs):
    
    bot = inputs["context"]
    input_payload = {
        "about_me": inputs["about_me"],
        "question": inputs["question"],
        "to_load_history": eval(inputs["history"]) if "history" in inputs else [],
    }
    response = bot.answer(**input_payload)

    return response


if __name__ == "__main__":
    fire.Fire(run_bot)