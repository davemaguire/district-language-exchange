import os
from twilio.rest import Client
import pendulum
import pandas as pd
import numpy as np
import yaml
from langchain import PromptTemplate, OpenAI, LLMChain
import math

# Organizer message
def get_organizer_df() -> pd.DataFrame:
    organizer_df: pd.DataFrame =  pd.read_csv('../organizer-info.csv') # need to read from GCS for cloud deployment
    return organizer_df

def increment_organizer_df(organizer_df: pd.DataFrame) -> None:
    # move next organizer to top of csv
    organizer_df.apply(np.roll, shift=1).to_csv('../organizer-info.csv', index=False) # need to write to GCS for cloud deployment

# get organizers & send message
def generate_organizer_message(organizer_df: pd.DataFrame) -> str:
    # Message
    next_date: str = pendulum.now().next(pendulum.SUNDAY).strftime('%m-%d')
    message_body: str = f'''
    This week's langX organizer is {organizer_df.organizer[0]}! Please schedule an event for {next_date}.
    Please follow these steps:
    1. Post in group chat that you are organizer this week
    2. Figure out a venue - you have decision making power (plan a backup for park events)
    3. Make FB event
    4. Make insta post
    5. Confirm at least 1 organizer will be there at 5 with stickers and pen. Post location pin for park events
    6. Make sure someone posts photos/videos as stories on instagram

    The next 3 organizers are:
    1. {organizer_df.organizer.iloc[-1]}
    2. {organizer_df.organizer.iloc[-2]}
    3. {organizer_df.organizer.iloc[-3]}
    '''

    return message_body

def ceil(n):
    return int(-1 * n // 1 * -1)

def set_secrets() -> None:
    # Load the config file
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get the value of the environment variable from the config
    os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
    os.environ['TWILIO_SID'] = config['TWILIO_SID']
    os.environ['TWILIO_AUTH_TOKEN'] = config['TWILIO_AUTH_TOKEN']
    os.environ['TWILIO_NUMBER'] = config['TWILIO_NUMBER']
    os.environ['ERROR_REPORT_NUMBER'] = config['ERROR_REPORT_NUMBER']
    print(os.environ['TWILIO_NUMBER'])

# ## Twilio Code
def send_organizer_text(organizer_df: pd.DataFrame, organizer_message: str, gpt_story:str) -> None:
    # This week's organizer
    organizer_phone: str = organizer_df.phone[0]

    # create twilio client
    client = Client(os.environ['TWILIO_SID'], os.environ['TWILIO_AUTH_TOKEN'])
    # Send message to all organizers
    # message = client.messages \
    #         .create(
    #             body='asfasdf',
    #             from_= f'+1{os.environ["TWILIO_NUMBER"]}',
    #             to= f'+1{os.environ["ERROR_REPORT_NUMBER"]}'
    #         )
    for org_num in organizer_df.phone:
        try:
            # if number of characters in message is greater than 1600, send two messages
            print(len(gpt_story))
            for message_chunk in range(0, ceil(len(gpt_story)/1600)):
                start_char: int = message_chunk * 1600
                end_char: int = start_char + 1600 - 1
                message = client.messages \
                        .create(
                            body=gpt_story[start_char:end_char],
                            from_= f'+1{os.environ["TWILIO_NUMBER"]}',
                            to=  f'+1{org_num}'
                        )          
            message = client.messages \
                        .create(
                            body=organizer_message,
                            from_= f'+1{os.environ["TWILIO_NUMBER"]}',
                            to= f'+1{org_num}'
                        )
        except Exception as e:
            failureMessage: str = f"""
            Failed to sent message to {org_num}. Error message:
            {e}
            """
            message = client.messages \
                        .create(
                            body=failureMessage,
                            from_= f'+1{os.environ["TWILIO_NUMBER"]}',
                            to= f'+1{os.environ["ERROR_REPORT_NUMBER"]}'
                        )

# def my_function():
#     print('hello from my_function')

# # openAI code
def generate_gpt_story(organizer_df) -> str:
    # This function will generate a story using the GPT-3 API
    story_types: list[str] = ["epic poem in the style of homer", 
                   "short story in the style of magical realism, emulating gabriel garcia marquez or jorge luis borges",
                   "picaresque story in the style of candide by voltaire or lazarillo de tormes",
                   "story in iambic pentameter in the style of william shakespeare",
                   "a story in the style of a gen Z texter or tik toker who doesn't capitalize or use punctuation",
                   "a whimsically imaginative poem in the style of dr. seuss"]
    story_type: str = np.random.choice(story_types)
    organizer_name: str = organizer_df.organizer[0]
    language: str = 'english' #np.random.choice(organizer_df.languages[0].split(','))
    interests: str = organizer_df.interests[0]
    template: str = """
    Write a {story_type}. The story should not be excessively long. The proganist is {organizer_name}, who will organizing a language exchange event. Tell the story in the {language} language.
    Incorporate {organizer_name}'s interests into the story. Do not simply regurgitate the list of their interests, but weave them into the story.
    Here are some of {organizer_name}'s interests: {interests}
    """
    prompt = PromptTemplate(template=template, input_variables=["story_type", "organizer_name", "language", "interests"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=1, model_name="gpt-4"), verbose=True)
    story: str = llm_chain.predict(story_type=story_type, organizer_name=organizer_name, language=language, interests=interests)
    return story



if __name__ == "__main__":
    """This runs when you execute '$ python3 mypackage/mymodule.py'"""
    set_secrets()
    organizer_df: pd.DataFrame = get_organizer_df()
    organizer_message = generate_organizer_message(organizer_df)
    gpt_story = generate_gpt_story(organizer_df)
    send_organizer_text(organizer_df, organizer_message, gpt_story) 
    increment_organizer_df(organizer_df)
