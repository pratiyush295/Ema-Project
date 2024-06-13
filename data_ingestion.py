
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to retrieve lecture notes
def get_lecture_notes(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    lectures = soup.find_all('div', class_='lecture-item')
    notes = {}
    for lecture in lectures:
        title = lecture.find('h2').text.strip()
        content = lecture.find('div', class_='content').text.strip()
        notes[title] = content
    return notes

# Stanford LLM lecture notes URL
lecture_notes_url = "https://stanford-cs324.github.io/winter2022/lectures/"
lecture_notes = get_lecture_notes(lecture_notes_url)

# Function to retrieve model architectures table
def get_model_architectures(url):
    df = pd.read_html(url, header=0)[0]
    return df

# GitHub table of model architectures URL
model_architectures_url = "https://github.com/Hannibal046/Awesome-LLM#milestone-papers"
model_architectures = get_model_architectures(model_architectures_url)
