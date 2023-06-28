import random
import json

import torch
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "AI Doctor"
symptom=[]
column_names = ['itching','skin rash','nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting',
 'vomiting', 'burning micturition',
 'spotting urination', 'fatigue','weight gain','anxiety','cold hands and feets','mood swings','weight loss','restlessness','lethargy','patches in throat','irregular sugar level',
 'cough', 'high fever',
 'sunken eyes', 'breathlessness',
 'sweating', 'dehydration',
 'indigestion', 'headache',
 'yellowish skin', 'dark urine',
 'nausea', 'loss of appetite',
 'pain behind the eyes', 'back pain',
 'constipation', 'abdominal pain',
 'diarrhoea', 'fever',
 'yellow urine', 'yellowing of eyes',
 'acute liver failure', 'fluid overload',
 'swelling of stomach', 'swelled lymph nodes',
 'malaise', 'blurred and distorted vision',
 'phlegm', 'throat irritation',
 'redness of eyes', 'sinus pressure',
 'runny nose', 'congestion',
 'chest pain', 'weakness in limbs',
 'fast heart rate', 'pain during bowel movements',
 'pain in anal region', 'bloody stool',
 'irritation in anus',
 'neck pain',
 'dizziness',
 'cramps',
 'bruising',
 'obesity',
 'swollen legs',
 'swollen blood vessels',
 'puffy face and eyes',
 'enlarged thyroid',
 'brittle nails',
 'swollen extremeties',
 'excessive hunger',
 'extra marital contacts',
 'drying and tingling lips',
 'slurred speech',
 'knee pain',
 'hip joint pain',
 'muscle weakness',
 'stiff neck',
 'swelling joints',
 'movement stiffness',
 'spinning movements',
 'loss of balance',
 'unsteadiness',
 'weakness of one body side',
 'loss of smell',
 'bladder discomfort',
 'foul smell of urine',
 'continuous feel of urine',
 'passage of gases',
 'internal itching',
 'toxic look (typhos)',
 'depression',
 'irritability',
 'muscle pain',
 'altered sensorium',
 'red spots over body',
 'belly pain',
 'abnormal menstruation',
 'dischromic patches',
 'watering from eyes',
 'increased appetite',
 'polyuria',
 'family history',
 'mucoid sputum',
 'rusty sputum',
 'lack of concentration',
 'visual disturbances',
 'receiving blood transfusion',
 'receiving unsterile injections',
 'coma',
 'stomach bleeding',
 'distention of abdomen',
 'history of alcohol consumption',
 'fluid overload.1',
 'blood in sputum',
 'prominent veins on calf',
 'palpitations',
 'painful walking',
 'pus filled pimples',
 'blackheads',
 'scurring',
 'skin peeling',
 'silver like dusting',
 'small dents in nails',
 'inflammatory nails',
 'blister',
 'red sore around nose',
 'yellow crust ooze']
df = pd.DataFrame(columns=column_names)
df.loc[0] = [0] * len(column_names)
disease_model = load_model('pred_model.h5',compile=False)
disease_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
disease=''
disease_names=['(vertigo) Paroymsal  Positional Vertigo','AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis',
 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemmorhoids(piles)', 'Drug Reaction', 'Fungal infection',
 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ',
 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthristis',
 'Paralysis (brain hemorrhage)', 'Peptic ulcer diseae', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins', 'hepatitis A']
dict={'Fungal infection':'A fungal infection, also called mycosis, is a skin disease caused by a fungus. There are '
                         'millions of species of fungi. They live in the dirt, on plants, on household surfaces, '
                         'and on your skin. Sometimes, they can lead to skin problems like rashes or bumps.',
      'Allergy':'Allergies, also known asÂ allergic diseases, are a number of conditions caused byÂ hypersensitivityÂ '
                'of the a immune systemÂ to typically harmless substances in the environment.Â These diseases '
                'includeÂ hay fever,Â food allergies,Â atopic dermatitis,Â allergic asthma, andÂ anaphylaxis.Â '
                'Symptoms may includeÂ red eyes, an itchy rash,Â sneezing, aÂ runny nose,Â shortness of breath, '
                'or swelling.Â Food intolerancesÂ andÂ food poisoningÂ are separate conditions.',
      'GERD':'Gastroesophageal reflux diseaseÂ (GERD), is aÂ chronicÂ condition in which stomach contents rise up '
             'into theÂ esophagus, resulting in either symptoms or complications.Â Symptoms include the taste of acid '
             'in the back of the mouth,Â heartburn,Â bad breath,Â chest pain, regurgitation, breathing problems, '
             'and wearing away of theÂ teeth.Complications includeÂ esophagitis,Â esophageal stricture, andÂ Barretts '
             'esophagus.',
      'Chronic cholestasis':'CholecystitisÂ isÂ inflammationÂ of theÂ gallbladder.Â Symptoms includeÂ right '
                            'upperÂ abdominal pain, nausea, vomiting, and occasionally fever.Â OftenÂ gallbladder '
                            'attacksÂ (biliary colic) precede acute cholecystitis.Â The pain lasts longer in '
                            'cholecystitis than in a typical gallbladder attack.Â Without appropriate treatment, '
                            'recurrent episodes of cholecystitis are common.Â Complications of acute cholecystitis '
                            'includeÂ gallstone pancreatitis,Â common bile duct stones, orÂ inflammation of the '
                            'common bile duct.',
      'Drug Reaction':'AnÂ adverse drug reactionÂ (ADR) is an injury caused by takingÂ medication.Â ADRs may occur '
                      'following a single dose or prolonged administration of aÂ drugÂ or result from the combination '
                      'of two or more drugs. The meaning of this term differs from the term "side effect" because '
                      'side effects can be beneficial as well as detrimental.Â The study of ADRs is the concern of '
                      'the field known asÂ pharmacovigilance. AnÂ adverse drug eventÂ (ADE) refers to any injury '
                      'occurring at the time a drug is used, whether or not it is identified as a cause of the '
                      'injury.Â An ADR is a special type of ADE in which a causative relationship can be shown. ADRs '
                      'are only one type of medication-related harm, as harm can also be caused by omitting to take '
                      'indicated medications.',
      'Peptic ulcer diseae':'Peptic ulcer diseaseÂ (PUD) is a break in the innerÂ lining of the stomach, the first '
                            'part of theÂ small intestine, or sometimes the lowerÂ esophagus.Â An ulcer in the '
                            'stomach is called aÂ gastric ulcer, while one in the first part of the intestines is '
                            'aÂ duodenal ulcer.Â The most common symptoms of a duodenal ulcer are waking at night '
                            'withÂ upper abdominal painÂ and upper abdominal pain that improves with eating.Â With a '
                            'gastric ulcer, the pain may worsen with eating.Â The pain is often described as '
                            'aÂ burningÂ or dull ache.Â Other symptoms includeÂ belching, vomiting, weight loss, '
                            'orÂ poor appetite.Â About a third of older people have no symptoms.Â Complications may '
                            'includeÂ bleeding,Â perforation, andÂ blockage of the stomach.Â Bleeding occurs in as '
                            'many as 15% of cases.',
      'AIDS':'Human immunodeficiency virus infection and acquired immunodeficiency syndromeÂ (HIV/AIDS) is a spectrum '
             'of conditions caused byÂ infectionÂ with theÂ human immunodeficiency virusÂ (HIV),'
             'Â aÂ retrovirus.Â Following initial infection a person may not notice any symptoms, or may experience a '
             'brief period ofÂ influenza-like illness.Â Typically, this is followed by a prolonged period with no '
             'symptoms.Â If the infection progresses, it interferes more with theÂ immune system, increasing the risk '
             'of developing common infections such asÂ tuberculosis, as well as otherÂ opportunistic infections, '
             'andÂ tumorsÂ which are otherwise rare in people who have normal immune function.Â These late symptoms '
             'of infection are referred to as acquired immunodeficiency syndrome (AIDS).Â This stage is often also '
             'associated withÂ unintended weight loss.',
      'Diabetes':'Diabetes mellitusÂ (DM), commonly known asÂ diabetes, is a group ofÂ metabolic '
                 'disordersÂ characterized by aÂ high blood sugarÂ level over a prolonged period of time.Â Symptoms '
                 'often includeÂ frequent urination,Â increased thirstÂ andÂ increased appetite.Â If left untreated, '
                 'diabetes can causeÂ many health complications.Â AcuteÂ complications can includeÂ diabetic '
                 'ketoacidosis,Â hyperosmolar hyperglycemic state, or death.Â Serious long-term complications '
                 'includeÂ cardiovascular disease,Â stroke,Â chronic kidney disease,Â foot ulcers,Â damage to the '
                 'nerves,Â damage to the eyesÂ andÂ cognitive impairment.',
      'Gastroenteritis':'GastroenteritisÂ is a medical term forÂ inflammationÂ of theÂ stomachÂ andÂ intestines. It '
                        'causesÂ diarrhea,Â vomitingÂ andÂ stomach pain. It usually happens because of infection by '
                        'aÂ virusÂ orÂ bacteria.',
      'Bronchial Asthma':'AsthmaÂ (orÂ Asthma bronchiale) is a disease that hurts theÂ airwaysÂ inside theÂ lungs. It '
                         'causes theÂ tissueÂ inside the airways toÂ swell. Asthma also causes the bands '
                         'ofÂ muscleÂ around the airways to become narrow. This makes it hard for enough air to pass '
                         'through and for the person to breathe normally. Asthma also '
                         'causesÂ mucus-makingÂ cellsÂ inside the airways to make more mucus than normal. This blocks '
                         'the airways, which are already very narrow during an asthma attack, and makes it even more '
                         'difficult to breathe.',
      'Hypertension':'HypertensionÂ (HTNÂ orÂ HT), also known asÂ high blood pressureÂ (HBP), '
                     'is aÂ long-termÂ medical conditionÂ in which theÂ blood pressureÂ in theÂ arteriesÂ is '
                     'persistently elevated.Â High blood pressure typically does not cause symptoms.Â Long-term high '
                     'blood pressure, however, is a major risk factor forÂ stroke,Â coronary artery disease,'
                     'Â heart failure,Â atrial fibrillation,Â peripheral arterial disease,Â vision loss,'
                     'Â chronic kidney disease, andÂ dementia.',
      'Migraine':'MigraineÂ (UK:Â /ËˆmiËÉ¡reÉªn/,Â US:Â /ËˆmaÉª-/)Â is aÂ primary headache disorderÂ characterized '
                 'by recurrentÂ headachesÂ that are moderate to severe.Â Typically, episodes affect one side of the '
                 'head, are pulsating in nature, and last from a few hours to three days.Â Associated symptoms may '
                 'includeÂ nausea,Â vomiting, andÂ sensitivity to light,Â sound, orÂ smell.Â The pain is generally '
                 'made worse by physical activity,Â although regular exercise may have prophylactic effects.Â Up to '
                 'one-third of people affected haveÂ aura: typically a short period of visual disturbance that '
                 'signals that the headache will soon occur.Â Occasionally, aura can occur with little or no headache '
                 'following.',
      'Cervical spondylosis':'SpondylosisÂ is the degeneration of theÂ vertebral columnÂ from any cause. In the more '
                             'narrow sense it refers to spinalÂ osteoarthritis, the age-related wear and tear of the '
                             'spinal column, which is the most common cause of spondylosis. The degenerative process '
                             'in osteoarthritis chiefly affects the vertebral bodies, theÂ neural foraminaÂ and '
                             'theÂ facet jointsÂ (facet syndrome). If severe, it may cause pressure on theÂ spinal '
                             'cordÂ orÂ nerve rootsÂ with subsequentÂ sensoryÂ orÂ motorÂ disturbances, '
                             'such asÂ pain,Â paresthesia,Â imbalance, andÂ muscle weaknessÂ in the limbs.',
      'Paralysis (brain hemorrhage)':'Intracerebral hemorrhageÂ (ICH), also known asÂ cerebral '
                                     'bleedÂ andÂ intraparenchymal bleed, is a sudden bleeding intoÂ the tissues of '
                                     'the brain, into itsÂ ventricles, or into both.Â It is one kind of bleeding '
                                     'within theÂ skullÂ and is one kind ofÂ stroke.',
      'Jaundice':'Jaundice, also known asÂ icterus, is a yellowish or greenish pigmentation of theÂ skinÂ andÂ whites '
                 'of the eyesÂ due toÂ high bilirubin levels.Â Jaundice in adults is typically a sign indicating the '
                 'presence of underlying diseases involving abnormalÂ hemeÂ metabolism,Â liver dysfunction, '
                 'orÂ biliary-tractÂ obstruction.Â The prevalence of jaundice in adults is rare, whileÂ jaundice in '
                 'babiesÂ is common, with an estimated 80% affected during their first week of life.Â The most '
                 'commonly associated symptoms of jaundice areÂ itchiness,Â paleÂ feces, andÂ dark urine.',
      'Malaria':'MalariaÂ is aÂ mosquito-borne infectious diseaseÂ that affects humans and other animals.Â Malaria '
                'causesÂ symptomsÂ that typically includeÂ fever,Â tiredness,Â vomiting, andÂ headaches.Â In severe '
                'cases, it can causeÂ yellow skin,Â seizures,Â coma, orÂ death.Â Symptoms usually begin ten to '
                'fifteen days after being bitten by an infectedÂ mosquito.Â If not properly treated, people may have '
                'recurrences of the disease months later.Â In those who have recently survived anÂ infection, '
                'reinfection usually causes milder symptoms.Â This partialÂ resistanceÂ disappears over months to '
                'years if the person has no continuing exposure to malaria.',
      'Chicken pox':'Chickenpox, also known asÂ varicella, is a highlyÂ contagiousÂ disease caused by the '
                    'initialÂ infectionÂ withÂ varicella zoster virusÂ (VZV).Â The disease results in a '
                    'characteristic skin rash that formsÂ small, itchy blisters, which eventually scab over.Â It '
                    'usually starts on the chest, back, and face.Â It then spreads to the rest of the body.Â Other '
                    'symptoms may includeÂ fever,Â tiredness, andÂ headaches.Â Symptoms usually last five to seven '
                    'days.Â Complications may occasionally includeÂ pneumonia,Â inflammation of the brain, '
                    'and bacterial skin infections.Â The disease is often more severe in adults than in '
                    'children.Â The incubation period is 10â€“21 days, 14â€“16 days, after which, a characteristic '
                    'rash appears.',
      'Dengue':'Dengue is a mosquito-borne, acute viral syndrome caused by any of the four serotypes of dengue virus '
               '(DENV). 1. In 2019, the World Health Organization designated dengue as one of the top 10 global '
               'health threats. 2. An estimated 50 million to 100 million symptomatic cases occur globally each year.',
      'Typhoid':'Typhoid fever, also known asÂ typhoid, is a disease caused byÂ SalmonellaÂ serotype Typhi '
                'bacteria.Â Symptoms may vary from mild to severe, and usually begin 6 to 30 days after '
                'exposure.Â Often there is a gradual onset of a highÂ feverÂ over several days.Â This is commonly '
                'accompanied by weakness,Â abdominal pain,Â constipation,Â headaches, and mild vomiting.Â Some people '
                'develop a skin rash withÂ rose colored spots.Â In severe cases, people may experience '
                'confusion.Â Without treatment, symptoms may last weeks or months.Â DiarrheaÂ is uncommon.Â Other '
                'people may carry the bacterium without being affected, but they are still able to spread the disease '
                'to others.Â Typhoid fever is a type ofÂ entericÂ fever, along withÂ paratyphoid fever.',
      'hepatitis A':'Hepatitis AÂ is an infectious disease of theÂ liverÂ caused byÂ Hepatovirus AÂ (HAV);Â it is a '
                    'type ofÂ viral hepatitis.Â Many cases have few or no symptoms, especially in the young.Â The '
                    'time between infection and symptoms, in those who develop them, is between two and six '
                    'weeks.Â When symptoms occur, they typically last eight weeks and may include nausea, vomiting, '
                    'diarrhea,Â jaundice, fever, and abdominal pain.Â Around 10â€“15% of people experience a '
                    'recurrence of symptoms during the six months after the initial infection.Â Acute liver '
                    'failureÂ may rarely occur, with this being more common in the elderly.',
      'Hepatitis B':'Hepatitis BÂ is anÂ infectious diseaseÂ caused by theÂ hepatitis B virusÂ (HBV) that affects '
                    'theÂ liver;Â it is a type ofÂ viral hepatitis.Â It can cause both acute andÂ chronic '
                    'infection.Â Many people have no symptoms during the initial infection.Â In acute infection, '
                    'some may develop a rapid onset of sickness with vomiting,Â yellowish skin,Â tiredness, '
                    'dark urine, andÂ abdominal pain.Â Often these symptoms last a few weeks and rarely does the '
                    'initial infection result in death.Â It may take 30 to 180 days for symptoms to begin.Â In those '
                    'who get infected around the time of birth 90% develop chronicÂ hepatitis BÂ while less than 10% '
                    'of those infected after the age of five do.Â Most of those with chronic disease have no '
                    'symptoms; however,Â cirrhosisÂ andÂ liver cancerÂ may eventually develop.Â Cirrhosis or liver '
                    'cancer occur in about 25% of those with chronic disease.',
      'Hepatitis C':'Hepatitis CÂ is anÂ infectious diseaseÂ caused by theÂ hepatitis C virusÂ (HCV) that primarily '
                    'affects theÂ liver;Â it is a type ofÂ viral hepatitis.Â During the initial infection people '
                    'often have mild or no symptoms.Â Occasionally a fever, dark urine, abdominal pain, andÂ yellow '
                    'tinged skinÂ occurs.Â The virus persists in the liver in about 75% to 85% of those initially '
                    'infected.Â Early on chronic infection typically has no symptoms.Â Over many years however, '
                    'it often leads toÂ liver diseaseÂ and occasionallyÂ cirrhosis.Â In some cases, those with '
                    'cirrhosis will develop serious complications such asÂ liver failure,Â liver cancer, orÂ dilated '
                    'blood vessels in the esophagusÂ andÂ stomach.',
      'Hepatitis D':'Hepatitis DÂ is a type ofÂ viralÂ hepatitisÂ caused by theÂ hepatitis delta virusÂ (HDV), '
                    'a smallÂ particleÂ that are alike toÂ viroidÂ andÂ virusoid.',
      'Hepatitis E':'Hepatitis E has mainly aÂ fecal-oralÂ transmission that is similar toÂ hepatitis A, '
                    'but the viruses are unrelated.',
      'Alcoholic hepatitis':'Alcoholic hepatitisÂ isÂ hepatitisÂ (inflammation of theÂ liver) due to excessive intake '
                            'ofÂ alcohol.Â Patients typically have a history of decades of heavy alcohol intake, '
                            'typically 8-10 drinks per day.Â It is usually found in association withÂ fatty liver, '
                            'an early stage ofÂ alcoholic liver disease, and may contribute to the progression of '
                            'fibrosis, leading toÂ cirrhosis. Symptoms may present acutely after a large amount of '
                            'alcoholic intake in a short time period, or after years of excess alcohol intake. Signs '
                            'and symptoms of alcoholic hepatitis includeÂ jaundiceÂ (yellowing of the skin and eyes),'
                            'Â ascitesÂ (fluid accumulation in theÂ abdominal cavity),Â fatigueÂ andÂ hepatic '
                            'encephalopathyÂ (brainÂ dysfunction due toÂ liver failure).Â Mild cases are '
                            'self-limiting, but severe cases have a high risk ofÂ death. Severe cases may be treated '
                            'withÂ glucocorticoids.',
      'Tuberculosis':'TuberculosisÂ (TB) is anÂ infectious diseaseÂ usually caused byÂ Mycobacterium tuberculosisÂ ('
                     'MTB)Â bacteria.Â Tuberculosis generally affects theÂ lungs, but can also affect other parts of '
                     'the body.Â Most infections show no symptoms, in which case it is known asÂ latent '
                     'tuberculosis.Â About 10% of latent infections progress to active disease which, '
                     'if left untreated, kills about half of those affected.Â The classic symptoms of active TB are a '
                     'chronicÂ coughÂ withÂ blood-containingÂ mucus,Â fever,Â night sweats, andÂ weight loss.Â It was '
                     'historically calledÂ consumptionÂ due to the weight loss.Â InfectionÂ of other organs can cause '
                     'a wide range of symptoms.',
      'Common Cold':'TheÂ common cold, also known simply as aÂ cold, is aÂ viralÂ infectious diseaseÂ of theÂ upper '
                    'respiratory tractÂ that primarily affects theÂ respiratory mucosaÂ of theÂ nose,Â throat,'
                    'Â sinuses, andÂ larynx.Â Signs and symptoms may appear less than two days after exposure to the '
                    'virus.Â These may includeÂ coughing,Â sore throat,Â runny nose,Â sneezing,Â headache, '
                    'andÂ fever.Â People usually recover in seven to ten days,Â but some symptoms may last up to '
                    'three weeks.Â Occasionally, those with otherÂ health problemsÂ may developÂ pneumonia.',
      'Pneumonia':'Pneumonia is an inflammatory condition of the lung primarily affecting the small air sacs '
                  'known as alveoli.Symptoms typically include some combination of productive or dry cough,'
                  'chest pain,fever and difficulty breathing.The severity of the condition is '
                  'variable.Pneumonia is usually caused by infection with viruses or bacteria, '
                  'and less commonly by other microorganisms. Identifying the responsible pathogen can be '
                  'difficult. Diagnosis is often based on symptoms and physical examination.Chest X-rays, '
                  'blood tests, and culture of the sputum may help confirm the diagnosis.The disease may be '
                  'classified by where it was acquired, such as community- or hospital-acquired or '
                  'healthcare-associated pneumonia.',
      'Dimorphic hemmorhoids(piles)':'HemorrhoidsÂ (Piles) are blood vessels located in the smooth muscles of the '
                                     'walls of the rectum and anus. They are a normal part of the anatomy and are '
                                     'located at the junction where small arteries merge into veins. They are '
                                     'cushioned by smooth muscles and connective tissue and are classified by where '
                                     'they are located in relationship to the pectinate line, the dividing point '
                                     'between the upper 2/3 and lower 1/3 of the anus. This is an important anatomic '
                                     'distinction because of the type of cells that lineÂ hemorrhoid, and the nerves '
                                     'that provide sensation.',
      'Heart attack':'A heart attack occurs when one or more of your coronary arteries becomes blocked. Over time, '
                     'a buildup of fatty deposits, including cholesterol, form substances called plaques, '
                     'which can narrow the arteries (atherosclerosis). This condition, called coronary artery '
                     'disease, causes most heart attacks.',
      'Varicose veins':'Varicose veins are twisted, enlarged veins. Any superficial vein may become varicosed, '
                       'but the veins most commonly affected are those in your legs. That is because standing and '
                       'walking upright increases the pressure in the veins of your lower body.',
      'Hypothyroidism':'Hypothyroidism (underactive thyroid) is a condition in which your thyroid gland does not '
                       'produce enough of certain crucial hormones. Hypothyroidism may not cause noticeable symptoms '
                       'in the early stages. Over time, untreated hypothyroidism can cause a number of health '
                       'problems, such as obesity, joint pain, infertility and heart disease.',
      'Hyperthyroidism':'Hyperthyroidism (overactive thyroid) occurs when your thyroid gland produces too much of the '
                        'hormone thyroxine. Hyperthyroidism can accelerate your bodys metabolism, '
                        'causing unintentional weight loss and a rapid or irregular heartbeat. Several treatments are '
                        'available for hyperthyroidism. Doctors use anti-thyroid medications and radioactive iodine '
                        'to slow the production of thyroid hormones. Sometimes, hyperthyroidism treatment involves '
                        'surgery to remove all or part of your thyroid gland.',
      'Osteoarthristis':'Osteoarthritis is the most common form of arthritis, affecting millions of people worldwide. '
                        'It occurs when the protective cartilage that cushions the ends of your bones wears down over '
                        'time. Although osteoarthritis can damage any joint, the disorder most commonly affects '
                        'joints in your hands, knees, hips and spine.',
      'Arthritis':'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of '
                  'arthritis are joint pain and stiffness, which typically worsen with age. The most common types of '
                  'arthritis are osteoarthritis and rheumatoid arthritis. Osteoarthritis causes cartilage â€” the '
                  'hard, slippery tissue that covers the ends of bones where they form a joint â€” to break down. '
                  'Rheumatoid arthritis is a disease in which the immune system attacks the joints, beginning with '
                  'the lining of joints.',
      '(vertigo) Paroymsal  Positional Vertigo':'Benign paroxysmal positional vertigo (BPPV) is one of the most '
                                                'common causes of vertigo â€” the sudden sensation that you are '
                                                'spinning or that the inside of your head is spinning. BPPVÂ causes '
                                                'brief episodes of mild to intense dizziness. It is usually triggered '
                                                'by specific changes in your heads position. This might occur when '
                                                'you tip your head up or down, when you lie down, or when you turn '
                                                'over or sit up in bed.',
      'Acne':'Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin '
             'cells. It causes whiteheads, blackheads or pimples. Acne is most common among teenagers, '
             'though it affects people of all ages. Effective acne treatments are available, but acne can be '
             'persistent. The pimples and bumps heal slowly, and when one begins to go away, others seem to crop up.',
      'Urinary tract infection':'A urinary tract infection (UTI) is an infection in any part of your urinary system '
                                'â€” your kidneys, ureters, bladder and urethra. Most infections involve the lower '
                                'urinary tract â€” the bladder and the urethra. Women are at greater risk of '
                                'developing aÂ UTIÂ than are men. Infection limited to your bladder can be painful '
                                'and annoying. However, serious consequences can occur if aÂ UTIÂ spreads to your '
                                'kidneys.',
      'Psoriasis':'Psoriasis is a skin disease that causes red, itchy scaly patches, most commonly on the knees, '
                  'elbows, trunk and scalp. Psoriasis is a common, long-term (chronic) disease with no cure. It tends '
                  'to go through cycles, flaring for a few weeks or months, then subsiding for a while or going into '
                  'remission. Treatments are available to help you manage symptoms. And you can incorporate lifestyle '
                  'habits and coping strategies to help you live better with psoriasis.',
      'Impetigo':'Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects '
                 'infants and young children. It usually appears as reddish sores on the face, especially around the '
                 'nose and mouth and on the hands and feet. Over about a week, the sores burst and develop '
                 'honey-colored crusts.',
      'Hypoglycemia':'Hypoglycemia is a condition in which your blood sugar (glucose) level is lower than normal. '
                     'Glucose is your bodys main energy source. Hypoglycemia is often related to diabetes treatment. '
                     'But other drugs and a variety of conditions â€” many rare â€” can cause low blood sugar in '
                     'people who dont have diabetes'}
dict_={'Fungal infection':'bath twice ,use detol or neem in bathing water,keep infected area dry,use clean cloths ',
      'GERD':'avoid fatty spicy food ,avoid lying down after eating, maintain healthy weight,exercise ',
      'Chronic cholestasis':'cold baths,anti itch medicine,consult doctor, eat healthy ',
      'Drug Reaction':'stop irritation,consult nearest hospital,stop taking drug,follow up ',
      'Peptic ulcer diseae':'avoid fatty spicy food , consume probiotic food, eliminate milk ,limit alcohol ',
      'AIDS':'avoid open cuts , wear ppe if possible, consult doctor, follow up',
      'Diabetes':'have balanced diet, exercise ,consult doctor , follow up',
      'Gastroenteritis':'stop eating solid food for while,	try taking small sips of water,	rest	,ease back into eating',
      'Bronchial Asthma':'switch to loose cloothing, take deep breaths, get away from trigger, seek help',
      'Hypertension':'meditation, salt baths, reduce stress	,get proper sleep',
      'Migraine':'meditation,reduce stress,use poloroid glasses in sun,consult doctor',
      'Cervical spondylosis':'use heating pad or cold pack,	exercise,	take otc pain reliver, consult doctor',
      'Paralysis (brain hemorrhage)':'massage, eat healthy,	exercise,	consult doctor',
      'Jaundice':'drink plenty of water	, consume milk thistle	,eat fruits and high fiberous food,	medication',
      'Malaria':'Consult nearest hospital,avoid oily food,avoid non veg food,keep mosquitos out ',
      'Chicken pox':'use neem in bathing ,	consume neem leaves	,take vaccine	,avoid public places',
      'Dengue':'drink papaya leaf juice	,avoid fatty spicy food,	keep mosquitos ,away	keep hydrated',
      'Typhoid':'eat high calorie vegitables ,	antiboitic therapy,	consult doctor	,medication ',
      'hepatitis A':'Consult nearest hospital, wash hands through ,	avoid fatty spicy food	,medication',
      'Hepatitis B':'consult nearest hospital,	vaccination	,eat healthy.	medication',
      'Hepatitis C':'Consult nearest hospital,	vaccination	,eat healthy,	medication',
      'Hepatitis D':'consult doctor,	medication,	eat healthy,	follow up',
      'Hepatitis E':'consult doctor, medication,	eat healthy	,follow up',
      'Alcoholic hepatitis':'stop alcohol consumption,	rest,	consult doctor,	medication',
      'Tuberculosis':'cover mouth,	consult doctor	,medication	,rest',
      'Common Cold':'drink vitamin c rich drinks,	take vapour,	avoid cold food	,keep fever in check',
      'Pneumonia':'consult doctor,	medication,	rest,	follow up',
      'Dimorphic hemmorhoids(piles)':'avoid fatty spicy food,	consume witch hazel	,warm bath with epsom salt,	consume alovera juice',
      'Heart attack':'call ambulance,	chew or swallow asprin	,keep calm',
      'Varicose veins':'lie down flat and raise the leg high	use oinments	use vein compression	dont stand still for long ',
      'Hypothyroidism':'reduce stress,exercise,eat healthy,get proper sleep ',
      'Hyperthyroidism':'eat healthy	massage	use lemon balm	take radioactive iodine treatment',
      'Osteoarthristis':'acetaminophen	consult nearest hospital	follow up	salt baths',
      'Arthritis':'Arthritis is the swelling and tenderness of one or more of your joints. The main symptoms of '
                  'arthritis are joint pain and stiffness, which typically worsen with age. The most common types of '
                  'arthritis are osteoarthritis and rheumatoid arthritis. Osteoarthritis causes cartilage â€” the '
                  'hard, slippery tissue that covers the ends of bones where they form a joint â€” to break down. '
                  'Rheumatoid arthritis is a disease in which the immune system attacks the joints, beginning with '
                  'the lining of joints.',
      '(vertigo) Paroymsal  Positional Vertigo':'lie down flat and raise the leg high,	use oinments,	use vein compression,	dont stand still for long',
      'Acne':'bath twice	,avoid fatty spicy food	drink plenty of water,	avoid too many products',
      'Urinary tract infection':'drink plenty of water,	increase vitamin c intake,	drink cranberry juice	take probiotics',
      'Psoriasis':'wash hands with warm soapy water ,stop bleeding using pressure,consult doctor,salt baths ',
      'Impetigo':'Impetigo (im-puh-TIE-go) is a common and highly contagious skin infection that mainly affects '
                 'infants and young children. It usually appears as reddish sores on the face, especially around the '
                 'nose and mouth and on the hands and feet. Over about a week, the sores burst and develop '
                 'honey-colored crusts.',
      'Hypoglycemia':'lie down on side,	check in pulse,	drink sugary drinks,	consult doctor'}
def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    print(tag, "msg-", msg)
    
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."



app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    if msg=="No" or msg=="Predict" or msg=="predict" or msg=="no":
        array = df.values
        array = np.asarray(array).astype(np.float32)
        predictions = disease_model.predict(array)
        predicted_class = np.argmax(predictions)
        d = disease_names[predicted_class]
        global disease
        disease=d
        text="You have {} , type 'Describe' to get full info , type 'Precations' if you need any".format(d)
        return text
    
    if msg=="Describe" or msg=="describe":
        if dict.get(disease) is not None:
            return dict.get(disease)
        else:
            return "Discription Not found :-("
    if msg=="Precations" or msg=="precations":
        if dict_.get(disease) is not None:
            return dict_.get(disease)
        else:
            return "Precations Not found :-("
    if msg in column_names:
        df.at[0,msg] = 1
    input = msg
    return get_response(input)


if __name__ == '__main__':
    app.run()


#0 0 0 0 0 