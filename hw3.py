import yaml

with open("rtvslo.yaml", "rt") as file:
    data = yaml.load(file, Loader=yaml.CLoader)

# print(data)

