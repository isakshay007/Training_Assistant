import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Training Schedule Generator")
st.markdown("Welcome to Training Schedule Generator! We create personalized workout plans based on your sport, experience level, and race date, ensuring you're perfectly prepared to crush your goals and perform at your best.")
st.markdown("            1) Name of the sports you are participating at.")
st.markdown("            2) Mention your experience level.")
st.markdown("            3) Mention the amount of time left until the race day.")
input = st.text_input(" Please enter the above details:",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role="Expert ATHLETIC COACH and PERSONAL TRAINER",
        prompt_persona=f"Your task is to DEVELOP a personalized training plan based on the INPUTS provided: the SPORT NAME, the athlete's EXPERIENCE LEVEL, and the RACE DATE.")
    prompt = f"""

You are an Expert ATHLETIC COACH specializing in creating CUSTOMIZED TRAINING SCHEDULES for athletes across MULTIPLE SPORTS. Your task is to DEVELOP a personalized training plan based on the INPUTS provided: the SPORT NAME, the athlete's EXPERIENCE LEVEL, and the RACE DATE.

Follow these steps to ensure a comprehensive training schedule:

1. ANALYZE the provided information about the specific SPORT for which you are devising the plan.ASSESS the athlete’s EXPERIENCE LEVEL to tailor the intensity and complexity of the training and IDENTIFY the time frame available until the RACE DATE to structure a progressive training regimen.

2. With the provided user information DESIGN a week-by-week or day by day TRAINING SCHEDULE that includes varied workouts, rest days, and milestones leading up to the race.

3. INCORPORATE exercises that focus on STRENGTH, ENDURANCE, TECHNIQUE, and RECOVERY appropriate for both the sport and experience level.

4. ENSURE there is a gradual build-up in intensity to peak at the right time before tapering down as race day approaches.

"""

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Generate"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)