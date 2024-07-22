from langchain.llms import OpenAI 
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain

import logging

logging.basicConfig(level=logging.INFO)

class ItineraryTemplate:
    def __init__(self):
        self.system_template = """
        You are a Q&A agent who helps older people to use a Android Cellphone.

        Retrieve a simple anwser with no more than 3 lines, no markdown notation.


        The user's request will be denoted by hashtags. Convert the
        user's request into a simple detaild list describing the actions 
        they should do.

     
        Return the answer as a bulleted list with clear instructions in how to do the actions.
        Be sure to be simples, because your audience is people with more than 60 years old.
        Your output must be the list and nothing else.
        """
        self.human_template = """
        ####{request}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["request"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )
        
   
class MappingTemplate:
    def __init__(self):
        self.system_template = """
        You are a Q&A agent who helps older people to use a Android Cellphone.

        Retrieve a simple anwser with no more than 3 lines, no markdown notation.

        For example:
        ###
        
        Para ligar o seu smartphone Android, pressione e segure o botão de energia até que a tela acenda. O botão de energia geralmente está localizado na lateral ou na parte superior do telefone. Se tiver dificuldades para encontrá-lo, consulte o manual do usuário do seu telefone para a localização exata.

        ###
        You Q&A agent system who converts the anwser into a list
        """
        self.human_template = """
        ####{agent_suggestion}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["agent_suggestion"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )
        
class CenterMapTemplate:
    def __init__(self):
        self.system_template = """
        You are an intelligent system that helps users visualize their travel plans.
        You receive a list of coordinates and must return the center of the map (aka. 
        the geodesic center of all the coordinates) and the zoom level that will allow.
        
        Retrieve a clean JSON object, no markdown notation.

        For example:
        ####
        {{
            "days": [
                {{ 
                "day": 1,
                "locations": [
                        {{"lat": 51.5014, "lon": -0.1419, "address": "The Mall, London SW1A 1AA", "name": "Buckingham Palace"}},
                        {{"lat": 51.5081, "lon": -0.0759, "address": "Tower Hill, London EC3N 4AB", "name": "Tower of London"}},
                        {{"lat": 51.5194, "lon": -0.1270, "address": "Great Russell St, Bloomsbury, London WC1B 3DG", "name": "British Museum"}},
                        {{"lat": 51.5145, "lon": -0.1444, "address": "Oxford St, London W1C 1JN", "name": "Oxford Street"}},
                        {{"lat": 51.5113, "lon": -0.1223, "address": "Covent Garden, London WC2E 8RF", "name": "Covent Garden"}},
                    ]
                }}, {{
                    "day": 2,
                    "locations": [
                        {{"lat": 51.4994, "lon": -0.1272, "address": "20 Deans Yd, Westminster, London SW1P 3PA", "name": "Westminster Abbey"}},
                        {{"lat": 51.5022, "lon": -0.1299, "address": "Clive Steps, King Charles St, London SW1A 2AQ", "name": "Churchill War Rooms"}},
                        {{"lat": 51.4966, "lon": -0.1764, "address": "Cromwell Rd, Kensington, London SW7 5BD", "name": "Natural History Museum"}},
                        {{"lat": 51.5055, "lon": -0.0754, "address": "Tower Bridge Rd, London SE1 2UP", "name": "Tower Bridge"}}
                    ]
                }}
            ]
        }}
        
        Output:
        {{
            "center": [51.5074, -0.1278],
            "zoom": 9
        }}
        """
        self.human_template = """
        ####{coordinates}####
        """
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(
            self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(
            self.human_template, input_variables=["agent_suggestion"]
        )
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )
        
class Agent:
    def __init__(
        self,
        open_ai_api_key,
        model= 'gpt-4o',
        temperature = 0, #não seja criativo
        verbose = True
    ):
        self.logger = logging.getLogger(__name__)
        if verbose:
            self.logger.setLevel(logging.INFO)
        
        self._openai_key = open_ai_api_key
        self.chat_model = ChatOpenAI(model=model,
                                     temperature=temperature,
                                     openai_api_key=self._openai_key)
        self.verbose = verbose
        
    def get_itinerary(self, request):
        itinerary_template = ItineraryTemplate()
        mapping_template = MappingTemplate()
        center_map_template = CenterMapTemplate()
        
        travel_agent = LLMChain(
            llm=self.chat_model,
            prompt=itinerary_template.chat_prompt,
            verbose=self.verbose,
            output_key='agent_suggestion'
        )
        coordinates_converter = LLMChain(
            llm=self.chat_model,
            prompt=mapping_template.chat_prompt,
            verbose=self.verbose,
            output_key='coordinates'
        )
        
        center_calculation = LLMChain(
            llm=self.chat_model,
            prompt=center_map_template.chat_prompt,
            verbose=self.verbose,
            output_key='center_info'
        )

        overall_chain = SequentialChain(
            chains=[travel_agent,
                    coordinates_converter,
                    center_calculation],
            input_variables=["request"],
            output_variables=["agent_suggestion",
                              "coordinates",
                              "center_info"],
            verbose=self.verbose
        )

        return overall_chain(
            {"request": request},
            return_only_outputs=True
        )