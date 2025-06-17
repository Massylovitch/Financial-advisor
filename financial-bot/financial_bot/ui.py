import gradio as gr
from financial_bot.bot_definition import FinancialBot


def predict(message, history, about_me):

    generate_kwargs = {
        "about_me": about_me,
        "question": message,
        "to_load_history": history,
    }

    yield bot.answer(**generate_kwargs)


bot = FinancialBot()

demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Ask me a financial question",
        label="Financial question",
        container=False,
        scale=7,
    ),
    additional_inputs=[
        gr.Textbox(
            "I am a student and I have some money that I want to invest.",
            label="About me",
        )
    ],
    title="Your Personal Financial Assistant",
    description="Ask me any financial or crypto market questions, and I will do my best to answer them.",
    theme="soft",
    type="messages",
    submit_btn=True,
    examples=[
        [
            "What's your opinion on investing in startup companies?",
            "I am a 30 year old graphic designer. I want to invest in something with potential for high returns.",
        ],
        [
            "What's your opinion on investing in AI-related companies?",
            "I'm a 25 year old entrepreneur interested in emerging technologies. \
             I'm willing to take calculated risks for potential high returns.",
        ],
        [
            "Do you think advancements in gene therapy are impacting biotech company valuations?",
            "I'm a 31 year old scientist. I'm curious about the potential of biotech investments.",
        ],
    ],
    cache_examples=False,
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
