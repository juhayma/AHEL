"""
Quick verification script for AHEL specialization scoring.
Run from project root:
    python analysis/test_specialization_scoring.py
This proves that changing specialization changes SPA and final readiness score.
"""

specialization_score_profiles = {
    'Data Science': {
        'required': ['Python','SQL','Data Analysis','Machine Learning','Power BI','Tableau','NLP','Big Data (Spark)'],
        'weights': {'Python':1.8,'SQL':1.5,'Data Analysis':1.7,'Machine Learning':1.8,'Power BI':1.3,'Tableau':1.2,'NLP':1.4,'Big Data (Spark)':1.3}
    },
    'Software Engineering': {
        'required': ['Java','C++','React','Git','Docker','Cloud (AWS/GCP)','SQL'],
        'weights': {'Java':1.6,'C++':1.4,'React':1.7,'Git':1.5,'Docker':1.4,'Cloud (AWS/GCP)':1.3,'SQL':1.1}
    },
    'Networks': {
        'required': ['Networking','Cybersecurity','Cloud (AWS/GCP)','Docker','Git','Python'],
        'weights': {'Networking':2.0,'Cybersecurity':1.5,'Cloud (AWS/GCP)':1.4,'Docker':1.2,'Git':1.1,'Python':1.1}
    },
    'AI & ML': {
        'required': ['Python','Machine Learning','Deep Learning','NLP','MLOps','Big Data (Spark)','Cloud (AWS/GCP)'],
        'weights': {'Python':1.7,'Machine Learning':2.0,'Deep Learning':1.8,'NLP':1.6,'MLOps':1.5,'Big Data (Spark)':1.3,'Cloud (AWS/GCP)':1.2}
    },
    'Cybersecurity': {
        'required': ['Cybersecurity','Networking','Python','SQL','Cloud (AWS/GCP)','Docker','Git'],
        'weights': {'Cybersecurity':2.0,'Networking':1.6,'Python':1.2,'SQL':1.1,'Cloud (AWS/GCP)':1.3,'Docker':1.2,'Git':1.1}
    }
}

def specialization_fit(selected_skills, specialization):
    if not specialization:
        return 30
    config = specialization_score_profiles.get(specialization, {'required': [], 'weights': {}})
    required = config['required']
    weights = config['weights']
    if not required:
        return 30
    possible = sum(weights.get(skill, 1) for skill in required)
    achieved = sum(weights.get(skill, 1) for skill in required if skill in selected_skills)
    if achieved == 0:
        return 10
    return round((achieved / possible) * 100)

def final_score(spa, pe=70, jda=75, sba=65, ed=72, cpr=70):
    sda = round(jda * .50 + sba * .20 + spa * .30)
    return round(pe*.25 + sda*.25 + ed*.20 + cpr*.15 + spa*.15)

skills = ['Python', 'SQL', 'Machine Learning']
print('Selected skills:', skills)
for spec in [None, 'Data Science', 'Software Engineering', 'Networks', 'AI & ML', 'Cybersecurity']:
    spa = specialization_fit(skills, spec)
    total = final_score(spa)
    print(f'{spec or "Not selected":22s} SPA={spa:3d} Final={total:3d}')
