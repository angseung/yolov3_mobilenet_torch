## xml-> to JSONM
import json
import os
import xmltodict


def xml_to_json(xml_dir, json_dir):
    for xml_file in os.listdir(xml_dir):
        print(xml_file)
        if "xml" not in xml_file:
            continue
        with open(f"{xml_dir}/{xml_file}", encoding="UTF-8") as fd:
            doc = xmltodict.parse(fd.read())
            dic = json.loads(json.dumps(doc))

        with open(f"../incoming/merge_json/{xml_file[:-3]}" + "json", "w") as fp:
            json.dump(dic, fp)
            print("save success")
