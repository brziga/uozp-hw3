import yaml

with open("rtvslo.yaml", "rt") as file:
    data = yaml.load(file, Loader=yaml.CLoader)

# print(data)

# l = 0
# for line in data:
#     l += len(line["gpt_keywords"])
# l /= len(data)
# print(l)
# # 20.22775...


# TF-IDF

# tf
#   - raw count of a term in a document (kolikokrat se pojavi keyword pri nekem dokumenti - načeloma 1 tukaj)
#   - denominator : koliko je vse skupaj kljucnih besed pri dokumentu
#   - sepravi basically 1/(st klucnih besed)

# idf
#   - N : st dokumentov - len(data)
#   - number of documents where the term appears

# plan:
#   - cez vse clanke - vrstice v data
#   - v novo strukturo:
#       - id clanka (njegov index v data)
#       - dict kjer so kljuci kljucne besede in vrednosti slovar z tf, idf in tf-idf
#       - najprej notri tf
#   - v locen pomozni dict se daje kljucne besede in steje, kolikokrat se pojavijo
#   - se en obhod cez data, zdaj se v novi strukturi doda se idf in poracuna tf-idf

