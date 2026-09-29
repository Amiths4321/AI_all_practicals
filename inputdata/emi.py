import glob
import json

def process_document():

          files = glob.glob("inputdata/*.jpg")

          print(files)

          for file in files:

                    if "pan" in file.lower():  
                    doc_type == "PAN"
                    print("PAN CARD")
                    elif "aadhar" in file.lower():
                    doc_type == "AADHAR"
                    print("AADHAR CARD")
                    elif "cheque" in file.lower():  
                    doc_type == "CHEQUE"
                    print("CHEQUE")
                    else:
                    doc_type == "OTHERS"
                    print("Unknown document type")

                    print(file, "->", doc_type)

                    result = {
                              "filename": file,
                              "document_type": doc_type,
                              "status": "success"
                    }

                    results.append(result)

                    print(results)

results = process_documents()

with open("results.json", "w") as f:
     json.dump(results, f, indent=4)

print("Processing completed")