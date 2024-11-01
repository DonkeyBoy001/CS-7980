import nltk
import csv
nltk.download('framenet_v17')
from nltk.corpus import framenet as fn

# Define keywords related to technology
tech_keywords = {"technology", "innovation", "science", "experiment", "development", "artificial intelligence"}

# Open CSV file and write header row
with open("tech_sentences.csv", "w", newline='', encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Frame", "Sentence", "Core Elements"])

    # Traversing all the frames
    for frame in fn.frames():
        frame_name = frame.name
        frame_elements = [e.name for e in frame.FE.values() if e.coreType == 'Core']
        
        # Traverse all the example sentences of the frame
        for s in fn.exemplars(frame=frame_name):
            sentence_text = s['text']
            
            # Determine whether the sentence contains scientific and technological keywords
            if any(keyword in sentence_text.lower() for keyword in tech_keywords):
                # Write the sentence, framework name, and core elements into a CSV file.
                csvwriter.writerow([frame_name, sentence_text, "+".join(frame_elements)])

print("Sentences related to the science and technology theme have been successfully collected and stored in the tech_sentences.csv file.")
