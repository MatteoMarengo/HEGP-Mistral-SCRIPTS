######
## Authors: LEVY Jarod & MARENGO Matteo
## Adapted from original work by Jean-Emmanuel Bibault & David J. Wu
## Date: January 2024
######

#######################################################################################################
# Import libraries
import dash
from dash import dcc, html, dash_table  # Updated import for dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import pandas as pd
import openai
import os
import dash_bootstrap_components as dbc
import subprocess
import time
import sys
from evaluation import compute_metric, scores_properties


from dotenv import load_dotenv
import datetime
load_dotenv()

openai.api_key = "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"
# import pdfkit
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.base import MIMEBase
# from email import encoders

external_stylesheets = [
    'https://maxcdn.bootstrapcdn.com/bootswatch/4.5.2/journal/bootstrap.min.css',
    dbc.themes.BOOTSTRAP
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
server = app.server

style = {
    'padding': '3.5em',
    'backgroundColor': '#FFFFFF',  
    'fontFamily': 'Arial, sans-serif'
}

#######################################################################################################
# Mapping for severity score of responses

severity_colors = {
    "None": '#008000',
    "Not at all": '#008000',
    "Never": '#008000',
    "Mild": '#90ee90',
    "A little bit": '#90ee90',
    "Rarely": '#90ee90',
    "Moderate": '#ffff00',
    "Somewhat": '#ffff00',
    "Occasionally": '#ffff00',
    "Severe": '#ffa500',
    "Quite a bit": '#ffa500',
    "Frequently": '#ffa500',
    "Very severe": '#ff0000',
    "Very much": '#ff0000',
    "Almost constantly": '#ff0000',
    "No": '#008000',
    "Yes": '#ff0000',
}

#######################################################################################################
# Define the layout of the application
def create_data_table():
    style_data_conditional = []

    for response, color in severity_colors.items():
        text_color = 'white' if color != '#ffff00' else 'black'
        style_data_conditional.append({
            'if': {
                'filter_query': '{{answer}} = "{}"'.format(response),
                'column_id': 'answer'
            },
            'backgroundColor': color,
            'color': text_color
        })

    return dash_table.DataTable(
        id='results_table',
        columns=[
            {'name': 'Question', 'id': 'question'},
            {'name': 'Answer', 'id': 'answer'},
        ],
        data=[],
        style_cell={
            'whiteSpace': 'normal',
            'height': 'auto',
            'textAlign': 'center',
            'border': 'none',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'border': 'none',
        },
        style_table={
            'margin': '0 auto',
            'width': '50%'
        },
        style_data_conditional=style_data_conditional
    )

# Define a function to create the timing data table
def create_timing_table():
    return dash_table.DataTable(
        id='timing_table',
        columns=[
            {"name": "Model", "id": "model"},
            {"name": "Time (seconds)", "id": "time"},
        ],
        data=[]  # Initial empty data
    )

current_date = datetime.date.today()

#######################################################################################################
app.layout = html.Div([
dcc.Markdown('# Prostate Radiotherapy Patient Symptom Intake Form - Comparaison Mistral7B / BioMistral7B / GPT-4 / Llama2-13B', style={'text-align': 'center'}),
    html.P([html.Br()]),
  dcc.Markdown('#### Please answer the following questions about your current symptoms'),
  dcc.Markdown('Each form must be carefully filled out, results will be sent to your physician'),
  dcc.Markdown("Today's date is: {}".format(current_date)),
  dcc.Markdown('#### **1: General Questions**'),
  dcc.Markdown('###### How many radiation treatments have you had? (ie 3, or I don\'t know)'),
dcc.Input(
    id='number_of_treatments',
    placeholder='Enter a value',
    type='text',
    value=''
),
  html.P([html.Br()]),
  dcc.Markdown('### **2: Symptom Questions**'),
  dcc.Markdown('For each of the following question I\'m going to ask you to grade your symptoms.'),
      dcc.Markdown("#### Fatigue"),
    dcc.Markdown(
        "###### In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST?"
    ),
    dcc.Dropdown(
        id="fatigue_severity",
        options=[
            {"label": "None", "value": "None"},
            {"label": "Mild", "value": "Mild"},
            {"label": "Moderate", "value": "Moderate"},
            {"label": "Severe", "value": "Severe"},
            {"label": "Very severe", "value": "Very severe"},
        ],
        value=None,
    ),

    dcc.Markdown(
        "###### In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities?"
    ),
    dcc.Dropdown(
        id="fatigue_interference",
        options=[
            {"label": "Not at all", "value": "Not at all"},
            {"label": "A little bit", "value": "A little bit"},
            {"label": "Somewhat", "value": "Somewhat"},
            {"label": "Quite a bit", "value": "Quite a bit"},
        ],
        value=None,
    ),
    html.P([html.Br()]),
  dcc.Markdown('#### Gas'),
    dcc.Markdown('###### In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)?'),
    dcc.Dropdown(
        id='gas',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'}
        ],
        value=None,
    ),
  dcc.Markdown('#### Diarrhea'),      
  dcc.Markdown('###### In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)?'),
    dcc.Dropdown(
    id='diarrhea_frequency',
    options=[
        {'label': 'Never', 'value': 'Never'},
        {'label': 'Rarely', 'value': 'Rarely'},
        {'label': 'Occasionally', 'value': 'Occasionally'},
        {'label': 'Frequently', 'value': 'Frequently'},
        {'label': 'Almost constantly', 'value': 'Almost constantly'}
    ],
    value=None,
    ),

    dcc.Markdown('#### Abdominal Pain'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)?'),
    dcc.Dropdown(
        id='abdominal_pain_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost Constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST?'),
    dcc.Dropdown(
        id='abdominal_pain_severity',
        options=[
            {'label': 'None', 'value': 'None'},
            {'label': 'Mild', 'value': 'Mild'},
            {'label': 'Moderate', 'value': 'Moderate'},
            {'label': 'Severe', 'value': 'Severe'},
            {'label': 'Very severe', 'value': 'Very severe'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='abdominal_pain_adl',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),
  html.P([html.Br()]),
  dcc.Markdown('Now let\'s discuss your urinary symptoms.'),
  dcc.Markdown('#### **3: Urinary Symptoms**'),
    dcc.Markdown('##### Painful Urination'),
    dcc.Markdown('###### In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST?'),
    dcc.Dropdown(
        id='painful_urination_severity',
        options=[
            {'label': 'None', 'value': 'None'},
            {'label': 'Mild', 'value': 'Mild'},
            {'label': 'Moderate', 'value': 'Moderate'},
            {'label': 'Severe', 'value': 'Severe'},
            {'label': 'Very severe', 'value': 'Very severe'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Urgency'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN?'),
    dcc.Dropdown(
        id='urinary_urgency_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_urgency_adl',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Frequency'),
    dcc.Markdown('###### In the last 7 days, were there times when you had to URINATE FREQUENTLY?'),
    dcc.Dropdown(
        id='urinary_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_frequency_interference',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Change in Usual Urine Color'),
    dcc.Markdown('###### In the last 7 days, did you have any URINE COLOR CHANGE?'),
    dcc.Dropdown(
        id='urine_color_change',
        options=[
            {'label': 'Yes', 'value': 'Yes'},
            {'label': 'No', 'value': 'No'}
        ],
        value=None,
    ),

    dcc.Markdown('##### Urinary Incontinence'),
    dcc.Markdown('###### In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)?'),
    dcc.Dropdown(
        id='urinary_incontinence_frequency',
        options=[
            {'label': 'Never', 'value': 'Never'},
            {'label': 'Rarely', 'value': 'Rarely'},
            {'label': 'Occasionally', 'value': 'Occasionally'},
            {'label': 'Frequently', 'value': 'Frequently'},
            {'label': 'Very much', 'value': 'Very much'},
            {'label': 'Almost constantly', 'value': 'Almost constantly'}
        ],
        value=None,
    ),

    dcc.Markdown('###### In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities?'),
    dcc.Dropdown(
        id='urinary_incontinence_interference',
        options=[
            {'label': 'Not at all', 'value': 'Not at all'},
            {'label': 'A little bit', 'value': 'A little bit'},
            {'label': 'Somewhat', 'value': 'Somewhat'},
            {'label': 'Quite a bit', 'value': 'Quite a bit'},
            {'label': 'Very much', 'value': 'Very much'}
        ],
        value=None,
    ),
    html.P([html.Br()]),
    dcc.Markdown("#### **4: Radiation Skin Reaction**"),
    dcc.Markdown(
        "###### In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST?"
    ),
    dcc.Dropdown(
        id="radiation_skin_reaction_severity",
        options=[
            {"label": "None", "value": "None"},
            {"label": "Mild", "value": "Mild"},
            {"label": "Moderate", "value": "Moderate"},
            {"label": "Severe", "value": "Severe"},
            {"label": "Very severe", "value": "Very severe"},
            {"label": "Not applicable", "value": "Not applicable"},
        ],
        value=None,
    ),
    html.P([html.Br()]),
    dcc.Markdown('#### **5: Last Question!**'),
    dcc.Markdown('###### Finally, do you have any other symptoms that you wish to report?'),
    dcc.Input(
        id='additional_symptoms',
        placeholder='Type here...',
        type='text',
        value=''),
    html.P([html.Br()]),
    dcc.Markdown(
        "###### Summarization Language"
    ),
    dcc.Dropdown(
        id="language",
        options=[
            {"label": "English", "value": "English"},
            {"label": "French", "value": "French"},
            {"label": "Portuguese", "value": "Portuguese"},
        ],
        value=None,
    ),
    html.Div(className="d-grid gap-2 d-flex justify-content-center", children=[
        dcc.Loading(id="loading", type="circle", children=[
            html.Button("Submit", id="submit_button", n_clicks=0, className="btn btn-lg btn-primary", style={"width": "200px"})
        ]),
    ]),
    html.Br(),
    html.Div([
        html.Div([
            html.Div('Mistral7B - LLM Library - Instruct-v0.1.Q4_0.gguf', className='card-header'),
            dcc.Loading(id="loading-summary-mistral7b", type="circle", children=[
                html.Div([
                    html.H4('Radiation Oncology Patient Symptom Summary using Mistral7B', className='card-title'),
                    html.P(id='summary_mistral7B', className='summary-container')
                ], className='card-body')
            ])
        ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
    ], className='summary-container mx-auto', style={'width': '60%'}),
#     html.Br(),
#     html.Div([
#         html.Div([
#             html.Div('Llama2-13B', className='card-header'),
#             dcc.Loading(id="loading-summary-llama2_13B", type="circle", children=[
#                 html.Div([
#                     html.H4('Radiation Oncology Patient Symptom Summary using Llama2-13B', className='card-title'),
#                     html.P(id='summary_llama2_13B', className='summary-container')
#                 ], className='card-body')
#             ])
#         ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
# ], className='summary-container mx-auto', style={'width': '60%'}),
    html.Br(),
    html.Div([
        html.Div([
            html.Div('Mistral7B - Ctransformers - Instruct-v0.1.Q5_K_M.gguf', className='card-header'),
            dcc.Loading(id="loading-summary-mistral7B-GGUF", type="circle", children=[
                html.Div([
                    html.H4('Radiation Oncology Patient Symptom Summary using Mistral7B GGUF', className='card-title'),
                    html.P(id='summary_mistral7B_GGUF', className='summary-container')
                ], className='card-body')
            ])
        ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
    ], className='summary-container mx-auto', style={'width': '60%'}),
    html.Br(),
    html.Div([
        html.Div([
            html.Div('BioMistral7B - Ctransformers - Q5_K_M.gguf', className='card-header'),
            dcc.Loading(id="loading-summary-BioMistral7B-GGUF", type="circle", children=[
                html.Div([
                    html.H4('Radiation Oncology Patient Symptom Summary using BioMistral7B GGUF', className='card-title'),
                    html.P(id='summary_BioMistral7B', className='summary-container')
                ], className='card-body')
            ])
        ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
    ], className='summary-container mx-auto', style={'width': '60%'}),
    html.Br(),
    html.Div([
        html.Div([
            html.Div('OpenAI', className='card-header'),
            dcc.Loading(id="loading-summary-gpt4", type="circle", children=[
                html.Div([
                    html.H4('Radiation Oncology Patient Symptom Summary using GPT4', className='card-title'),
                    html.P(id='summary_gpt4', className='summary-container')
                ], className='card-body')
            ])
        ], className='card border-primary mb-3', style={'max-width': '60rem', 'margin': '3 auto'})
    ], className='summary-container mx-auto', style={'width': '60%'}),
    html.Br(),
    html.Div([
        dcc.Markdown('### Survey Results')
    ], style={'textAlign': 'center'}),
    create_data_table(),
    html.Br(),
    html.Div([
        dcc.Markdown('### Summary Generation Timing')
    ], style = {'textAlign': 'center'}),
    create_timing_table(),
    html.Br(),
    dcc.Markdown("""#### **Disclaimer**"""),
    'This application does not store any of the data provided. It is used with the Mistral7B in local so data is not shared.',
    html.Br(),
    html.Br(),
    'All information contained on this application is for general informational purposes only and does not constitute medical or other professional advice. It should not be used for medical purpose. The information you find on this application is provided as is and to the extent permitted by law, the creators disclaim all other warranties of any kind, whether express, implied, statutory or otherwise. In no event will the authors or its affiliates be liable for any damages whatsoever, including without limitation, direct, indirect, incidental, special, exemplary, punitive or consequential damages, arising out of the use, inability to use or the results of use of this application, any websites linked to this application or the materials or information or services contained at any or all such websites, whether based on warranty, contract, tort or any other legal theory and whether or not advised of the possibility of such damages. If your use of the materials, information or services from this application or any website linked to this application results in the need for servicing, repair or correction of equipment or data, you assume all costs thereof.)',
    html.Br(),
    html.Br(),
    dcc.Markdown("""#### **About**"""),
    'Created by David J. Wu & Jean-Emmanuel Bibault. Adapted by Jarod Lévy & Matteo Marengo - 2023/2024',
        ], style=style)

@app.callback(
    [Output('summary_gpt4', 'children'),
     Output('summary_mistral7B', 'children'),
     Output('summary_mistral7B_GGUF', 'children'),
     Output('summary_BioMistral7B', 'children'),
     Output('results_table', 'data'),
     Output('timing_table', 'data')],
    [Input('submit_button', 'n_clicks')],
    [State('number_of_treatments', 'value'),
    State('gas', 'value'),
    State('diarrhea_frequency', 'value'),
    State('abdominal_pain_frequency', 'value'),
    State('abdominal_pain_severity', 'value'),
    State('abdominal_pain_adl', 'value'),
    State('painful_urination_severity', 'value'),
    State('urinary_urgency_frequency', 'value'),
    State('urinary_urgency_adl', 'value'),
    State('urinary_frequency', 'value'),
    State('urinary_frequency_interference', 'value'),
    State('urine_color_change', 'value'),
    State('urinary_incontinence_frequency', 'value'),
    State('urinary_incontinence_interference', 'value'),
    State('radiation_skin_reaction_severity', 'value'),
    State('fatigue_severity', 'value'),
    State('fatigue_interference', 'value'),
    State('additional_symptoms', 'value'),
    State('language', 'value')]
)

#######################################################################################################
def update_table_results(n_clicks, *responses):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, [], []
        #return None, []

    questions = [
        'Number of Radiation treatments',
        'Increased passing of gas',
        'Diarrhea frequency',
        'Abdominal pain frequency',
        'Abdominal pain severity',
        'Abdominal pain with ADL',
        'Painful urination severity',
        'Urinary urgency frequency',
        'Urinary urgency with ADL',
        'Urinary frequency',
        'Urinary frequency with ADL',
        'Urine color change',
        'Urinary incontinence frequency',
        'Urinary incontinence with ADL',
        'Radiation skin reaction severity',
        'Fatigue severity',
        'Fatigue with ADL',
        'Additional symptoms',
    ]

    data = [{'question': question, 'answer': response} for question, response in zip(questions, responses)]
    language = responses[-1]
     # Generate summaries for both models
    summary_gpt4,time_gpt4 = summarize_table_gpt4(data, language)
    summary_mistral7B, time_mistral7B = summarize_table_mistral7B(data, language)  # Assume you adapt your summarization function accordingly
    #summary_llama2_13B = summarize_table_llama2_13B(data, language)  # Assume you adapt your summarization function accordingly
    summary_mistral7B_GGUF, time_mistral7B_GGUF = summarize_table_mistral7B_GGUF(data, language)  # Assume you adapt your summarization function accordingly
    summary_BioMistral7B, time_BioMistral7B = summarize_table_mistral7B_BIO(data, language)

    timing_data = [
        {'model': 'GPT-4', 'time': time_gpt4},
        {'model': 'Mistral7B', 'time': time_mistral7B},
        {'model': 'Mistral7B-GGUF', 'time': time_mistral7B_GGUF},
        {'model': 'BioMistral7B-GGUF', 'time': time_BioMistral7B}
    ]
    
    return summary_gpt4, summary_mistral7B, summary_mistral7B_GGUF, summary_BioMistral7B, data, timing_data

#######################################################################################################
def summarize_table_mistral7B(data, language):
    start_time = time.time() # Start timing

    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
    }]
    
    for row in data:
        messages.append({
            'role': 'user',
            'content': f"{row['question']}: {row['answer']}"
        })
    
    messages.append({
        'role': 'assistant',
        'content': "Summary:"
    })
   
    command = [
        "llm", "-m", "mistral-7b-instruct-v0", 
        str(messages)
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = process.communicate()
    output = stdout.decode('utf-8')
    # summary = response.choices[0].message.content.strip()
    summary = output 
    end_time = time.time() # End timing
    summary_time = end_time - start_time # Calculate time

    return summary, summary_time

#######################################################################################################
from ctransformers import AutoModelForCausalLM, AutoConfig
import os

def summarize_table_mistral7B_GGUF(data, language):
    start_time = time.time() # Start timing
    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
    }]
    
    for row in data:
        messages.append({
            'role': 'user',
            'content': f"{row['question']}: {row['answer']}"
        })
    
    messages.append({
        'role': 'assistant',
        'content': "Summary:"
    })

    model_name= "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    model_path = r"D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/GGUF-Mistral7B/mistral-7b-instruct-v0.1.Q5_K_M.gguf"

    # Create the llm
    config = AutoConfig.from_pretrained(model_name)
    # config.max_new_tokens = 2048
    config.config.context_length = 4096
    llm = AutoModelForCausalLM.from_pretrained(model_name,
                                                model_file=model_path,
                                                model_type="mistral",
                                                temperature=0.7,
                                                gpu_layers=0,
                                                stream=True,
                                                threads=int(os.cpu_count()),
                                                max_new_tokens=200,
                                                config=config)
    # text = "Q1. How many radiation treatments have you had? It’s okay if you don’t know. A1. 3 Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? A2. Severe Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? A3. Quite a bit Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? A4. Yes Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? A5. Frequently Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? A6. Frequently Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? A7. Severe Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? A8. Very much Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? A9. Moderate Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? A10. Rarely Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? A11. A little bit Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? A12. Frequently Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? A13. Quite a bit Q14. In the last 7 days, did you have any URINE COLOR CHANGE? A14. No Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? A15. Occasionally Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? A16. Somewhat Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? A17. Very Severe Q18. Finally, do you have any other symptoms that you wish to report? A18. Slight nausea and dizziness. You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
    prompt = f"[INST]{str(messages)}[/INST]"

    # Generate the summary
    summary = llm(prompt=prompt)

    end_time = time.time() # End timing
    summary_time = end_time - start_time # Calculate time

    return summary, summary_time

from llama_cpp import Llama

# def summarize_table_mistral7B_BIO(data, language):
#     start_time = time.time() # Start timing
#     messages = [{
#         'role': 'system',
#         'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
#     }]
    
#     for row in data:
#         messages.append({
#             'role': 'user',
#             'content': f"{row['question']}: {row['answer']}"
#         })
    
#     messages.append({
#         'role': 'assistant',
#         'content': "Summary:"
#     })

#     # model_name= "MaziyarPanahi/BioMistral-7B-GGUF"
#     model_path = r"D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/BioMistral-7B.Q5_K_M.gguf"
#     # Create the llm
#     # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
#     llm = Llama(
#     model_path=model_path,  # Download the model file first
#     n_ctx=4000,  # The max sequence length to use - note that longer sequence lengths require much more resources
#     n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
#     n_gpu_layers=0         # The number of layers to offload to GPU, if you have GPU acceleration available
#     )

#     # Simple inference example
#     output = llm(
#         "<|im_start|>system{system_message}<|im_end|><|im_start|>user{prompt}<|im_end|><|im_start|>assistant", #
#         max_tokens=200,  # Generate up to 512 tokens
#         stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#         echo=True        # Whether to echo the prompt
#     )

#     # Chat Completion API

#     llm = Llama(model_path=model_path, chat_format="mistral")  # Set chat_format according to the model you are using
#     llm.create_chat_completion(
#         messages = [
#             {"role": "system", "content": "You are a story writing assistant."},
#             {
#                 "role": "user",
#                 "content": "Write a story about llamas."
#             }
#         ]
#     )

#     end_time = time.time() # End timing
#     summary_time = end_time - start_time # Calculate time

#     return summary, summary_time

def summarize_table_mistral7B_BIO(data, language):
    start_time = time.time() # Start timing
    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
    }]
    
    for row in data:
        messages.append({
            'role': 'user',
            'content': f"{row['question']}: {row['answer']}"
        })
    
    messages.append({
        'role': 'assistant',
        'content': "Summary:"
    })

    model_name= "BioMistral/BioMistral-7B-GGUF"
    model_path = r"D:/OneDrive/Documents/MVA-ENS-2023-2024/S1/HEGP/HEGP-Mistral-SCRIPTS/ggml-model-Q5_K_M.gguf"
    # Create the llm
    config = AutoConfig.from_pretrained(model_name)
    # config.max_new_tokens = 2048
    config.config.context_length = 4096
    llm = AutoModelForCausalLM.from_pretrained(model_name,
                                                model_file=model_path,
                                                model_type="mistral",
                                                temperature=0.7,
                                                gpu_layers=0,
                                                stream=True,
                                                threads=int(os.cpu_count()),
                                                max_new_tokens=200,
                                                config=config)
    # text = "Q1. How many radiation treatments have you had? It’s okay if you don’t know. A1. 3 Q2. In the last 7 days, what was the SEVERITY of your FATIGUE, TIREDNESS, OR LACK OF ENERGY at its WORST? A2. Severe Q3. In the last 7 days, how much did FATIGUE, TIREDNESS, OR LACK OF ENERGY INTERFERE with your usual or daily activities? A3. Quite a bit Q4. In the last 7 days, did you have any INCREASED PASSING OF GAS (FLATULENCE)? A4. Yes Q5. In the last 7 days, how OFTEN did you have LOOSE OR WATERY STOOLS (DIARRHEA)? A5. Frequently Q6. In the last 7 days, how OFTEN did you have PAIN IN THE ABDOMEN (BELLY AREA)? A6. Frequently Q7. In the last 7 days, what was the SEVERITY of your PAIN IN THE ABDOMEN (BELLY AREA) at its WORST? A7. Severe Q8. In the last 7 days, how much did PAIN IN THE ABDOMEN (BELLY AREA) INTERFERE with your usual or daily activities? A8. Very much Q9. In the last 7 days, what was the SEVERITY of your PAIN OR BURNING WITH URINATION at its WORST? A9. Moderate Q10. In the last 7 days, how OFTEN did you feel an URGE TO URINATE ALL OF A SUDDEN? A10. Rarely Q11. In the last 7 days, how much did SUDDEN URGES TO URINATE INTERFERE with your usual or daily activities? A11. A little bit Q12. In the last 7 days, were there times when you had to URINATE FREQUENTLY? A12. Frequently Q13. In the last 7 days, how much did FREQUENT URINATION INTERFERE with your usual or daily activities? A13. Quite a bit Q14. In the last 7 days, did you have any URINE COLOR CHANGE? A14. No Q15. In the last 7 days, how OFTEN did you have LOSS OF CONTROL OF URINE (LEAKAGE)? A15. Occasionally Q16. In the last 7 days, how much did LOSS OF CONTROL OF URINE (LEAKAGE) INTERFERE with your usual or daily activities? A16. Somewhat Q17. In the last 7 days, what was the SEVERITY of your SKIN BURNS FROM RADIATION at their WORST? A17. Very Severe Q18. Finally, do you have any other symptoms that you wish to report? A18. Slight nausea and dizziness. You are an experienced radiation oncologist physician. You are provided this list of questions and answers about patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put the most important symptoms first. Provide the summarization in the english language. Example: This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash. "
    prompt = f"[INST]{str(messages)}[/INST]"

    # Generate the summary
    summary = llm(prompt=prompt)

    end_time = time.time() # End timing
    summary_time = end_time - start_time # Calculate time

    return summary, summary_time

# #######################################################################################################
## WARNING: 6.86 GB download, needs 16GB RAM to run
# def summarize_table_llama2_13B(data, language):
#     messages = [{
#         'role': 'system',
#         'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
#     }]
    
#     for row in data:
#         messages.append({
#             'role': 'user',
#             'content': f"{row['question']}: {row['answer']}"
#         })
    
#     messages.append({
#         'role': 'assistant',
#         'content': "Summary:"
#     })
   
#     command = [
#         "llm", "-m", "nous-hermes-llama2-13b", 
#         str(messages)
#     ]

#     process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

#     stdout, stderr = process.communicate()
#     output = stdout.decode('utf-8')
#     # summary = response.choices[0].message.content.strip()
#     summary = output 
#     return summary

#######################################################################################################
def summarize_table_gpt4(data, language):
    start_time = time.time() # Start timing
    messages = [{
        'role': 'system',
        'content': f"You are an experienced radiation oncologist physician. You are provided this table of patient symptoms during their weekly follow-up visit during radiotherapy. Please summarize the following data into two sentences of natural language for your physician colleagues. Please put most important symptoms first. Provide the summarization in the {language} language. English Example - This patient with 7 radiation treatments is having severe abdominal pain, moderately affecting activities of daily living. Other symptoms include occasional diarrhea, mild rash.:"
    }]
    
    for row in data:
        messages.append({
            'role': 'user',
            'content': f"{row['question']}: {row['answer']}"
        })
    
    messages.append({
        'role': 'assistant',
        'content': "Summary:"
    })

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        n=1,
        stop=None,
        temperature=0.2,
    )
    
    summary = response.choices[0].message.content.strip() 
    end_time = time.time() # End timing
    summary_time = end_time - start_time # Calculate time
    return summary,summary_time

#######################################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)
