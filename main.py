import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# AVAILABLE WORK YEARS 2020 - 2023
def int_to_company_location(number):
    countrys_indexes = ['Spain', 'United States', 'Canada', 'Germany', 'United Kingdom', 'Nigeria', 'India',
                 'Hong Kong', 'Netherlands', 'Switzerland', 'Central African Republic', 'France',
                 'Finland', 'Ukraine', 'Ireland', 'Israel', 'Ghana', 'Colombia', 'Singapore',
                 'Australia', 'Sweden', 'Slovenia', 'Mexico', 'Brazil', 'Portugal', 'Russia',
                 'Thailand', 'Croatia', 'Argentina', 'Lithuania', 'American Samoa', 'Costa Rica',
                 'Iran', 'Bahamas', 'Hungary', 'Austria', 'Slovakia', 'Czech Republic', 'Turkey',
                 'Puerto Rico', 'Denmark', 'Bolivia', 'Philippines', 'Belgium', 'Indonesia',
                 'Egypt', 'United Arab Emirates', 'Luxembourg', 'Malaysia', 'Honduras', 'Japan',
                 'Algeria', 'Iraq', 'China', 'New Zealand', 'Chile', 'Moldova', 'Malta']

    
    if 0 <= number < len(countrys_indexes):
        return countrys_indexes[number - 1] # I forgor to include number 0 in company_location
    else:
        return "Invalid index" 
def int_to_experience_level(number):
    levels = ['Senior', 'Middle', 'Entry level','Executive'] 
    if 0 <= number < len(levels):
        return levels[number] 
    else:
        return "Invalid index" 
def int_to_work_type(number):
    types = ['Full time', 'Contractor', 'Freelance', 'Part time']
    if 0 <= number < len(types):
        return types[number] 
    else:
        return "Invalid index" 

def int_to_job_title(number):
    jobs = ['Principal Data Scientist', 'ML Engineer', 'Data Scientist', 'Applied Scientist', 'Data Analyst', 'Data Modeler', 'Research Engineer', 'Analytics Engineer', 'Business Intelligence Engineer', 'Machine Learning Engineer', 'Data Strategist', 'Data Engineer', 'Computer Vision Engineer', 'Data Quality Analyst', 'Compliance Data Analyst', 'Data Architect', 'Applied Machine Learning Engineer', 'AI Developer', 'Research Scientist', 'Data Analytics Manager', 'Business Data Analyst', 'Applied Data Scientist', 'Staff Data Analyst', 'ETL Engineer', 'Data DevOps Engineer', 'Head of Data', 'Data Science Manager', 'Data Manager', 'Machine Learning Researcher', 'Big Data Engineer', 'Data Specialist', 'Lead Data Analyst', 'BI Data Engineer', 'Director of Data Science', 'Machine Learning Scientist', 'MLOps Engineer', 'AI Scientist', 'Autonomous Vehicle Technician', 'Applied Machine Learning Scientist', 'Lead Data Scientist', 'Cloud Database Engineer', 'Financial Data Analyst', 'Data Infrastructure Engineer', 'Software Data Engineer', 'AI Programmer', 'Data Operations Engineer', 'BI Developer', 'Data Science Lead', 'Deep Learning Researcher', 'BI Analyst', 'Data Science Consultant', 'Data Analytics Specialist', 'Machine Learning Infrastructure Engineer', 'BI Data Analyst', 'Head of Data Science', 'Insight Analyst', 'Deep Learning Engineer', 'Machine Learning Software Engineer', 'Big Data Architect', 'Product Data Analyst', 'Computer Vision Software Engineer', 'Azure Data Engineer', 'Marketing Data Engineer', 'Data Analytics Lead', 'Data Lead', 'Data Science Engineer', 'Machine Learning Research Engineer', 'NLP Engineer', 'Manager Data Management', 'Machine Learning Developer', '3D Computer Vision Researcher', 'Principal Machine Learning Engineer', 'Data Analytics Engineer', 'Data Analytics Consultant', 'Data Management Specialist', 'Data Science Tech Lead', 'Data Scientist Lead', 'Cloud Data Engineer', 'Data Operations Analyst', 'Marketing Data Analyst', 'Power BI Developer', 'Product Data Scientist', 'Principal Data Architect', 'Machine Learning Manager', 'Lead Machine Learning Engineer', 'ETL Developer', 'Cloud Data Architect', 'Lead Data Engineer', 'Head of Machine Learning', 'Principal Data Analyst', 'Principal Data Engineer', 'Staff Data Scientist', 'Finance Data Analyst']
    if 0 <= number < len(jobs):
        return jobs[number] 
    else:
        return "Invalid index"   
def int_to_company_size(number):
    sizes = ['Large', 'Small', 'Medium']
    if 0 <= number < len(sizes):
        return sizes[number] 
    else:
        return "Invalid index" 

file = pd.read_csv("./ds_salaries.csv")



# 6 inputs 1 output(salary)
# output salary_in_usd
# inputs work_year, experience_level, employment_type, job_title, company_location, company_size
class Model(nn.Module):
    def __init__(self, inputs=6,h1=20,h2=25,output_features=1): # fc = fully connected layer(connectin layers)
        super().__init__()
        self.fc1 = nn.Linear(inputs,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.out = nn.Linear(h2,output_features)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x
    

torch.manual_seed(41)

model = Model()


file.to_csv("./ds_salaries.csv",index=False)
print(file.columns)