"""
SCRIPT 1:
This script uses lxml + xpath and sometimes regular expressions to extract the syntactic relations from the parsed xml files.
It is strongly recommended to consult the Universal Dependencies-Guidelines for further information: http://universaldependencies.org/u/dep/all.html 
The parsing is not free from errors and in addition to that the extracted numbers are not 100% error-free either, because it was not always possible to find
an extracting rule that covers all irregularities that happen especially due to the nature of spoken language. This entails, that the distinction between
independent and dependent clauses is sometimes unclear.


NOTE: This script is probably the most original script in this repository, because every extracting rule was created by myself.
"""

import os
import glob
import pandas as pd
import lxml.etree
import re

SAMPLE = "sample_1930-2010"
FILE_DIR = "XXX\\filmgenre_classification\\5_sample\\" + SAMPLE + "\\xml"
SYN_CSV = "XXX\\filmgenre_classification\\7_features\\" + SAMPLE + "_syn-features.csv"

def get_features(FILE_DIR):
    """
    Coordinating function: Loops over all XML-Files and collects every wanted syntactic metric.
    Data is saved in a nested dictionary {file1: {measure1: 5, measure2: 10...} file2:...}, so that it can be transformed to a DataFrame later on.
    """
    syn_features = {}
    for xml_file in glob.glob(FILE_DIR + "\\*.xml"):
        base = os.path.basename(xml_file)
        filename, extension = base.split(".")
        print("Processing file", filename)

        xml_tree = lxml.etree.parse(xml_file)
        words = count_words(xml_tree)
        sentences = count_sentences(xml_tree)
        total_clauses, dependent_clauses, adjectival_clauses_per_1000_words = count_clauses(xml_tree, words)
        passives = count_passives(xml_tree)
        number_of_noun_phrases, number_of_complex_nominals, mean_length_of_noun_phrases = count_noun_phrases_and_complex_nominals(xml_tree)
        #print("sen", sentences)
        #print("claus", total_clauses)
        #print("dep", dependent_clauses)
        """
        Key:
        1. MLS: Mean length of sentence
        2. MLC: Mean length of clause
        3. MLNP: Mean leangth of noun phrase
        4. SCR: Sentence complexity ratio
        5. DPPC: Dependent clauses per clause
        6. ADJCL: Number of adjectival clauses per 1000 words
        7. PASSPS: Passives per sentence
        8. NPPS: Mean number of noun phrases per sentence
        9. CNPS: Mean number of complex nominals per sentence
        10. CNPC: Mean number of complex nominals per clause
        """
        syn_features[filename] = {"MLS": words / sentences, "MLC": words / total_clauses, "MLNP": mean_length_of_noun_phrases, "SCR": total_clauses / sentences, "DPPC": dependent_clauses / total_clauses, 
        "ADJCL" : adjectival_clauses_per_1000_words, "PASSPS": passives / sentences, 
        "NPPS": number_of_noun_phrases / sentences, "CNPS": number_of_complex_nominals / sentences, "CNPC" : number_of_complex_nominals / total_clauses}
    return syn_features

def count_words(xml_tree):
    """
    Count all occurring words, which is achieved by getting all w-elements and substracting every punctuation and symbol
    """
    number_of_w_elements = xml_tree.xpath("count (//w)")
    number_of_punctuation = xml_tree.xpath("count (//w[@xpos='PUNCT'])")
    number_of_symbols = xml_tree.xpath("count (//w[xpos='SYM'])")
    number_of_words = number_of_w_elements - number_of_punctuation -number_of_symbols
    return int(number_of_words)

def count_sentences(xml_tree):
    """
    Every s-element with at least one word corresponds to a sentence. Therefore sentences consisting solely of punctuations and symbols are deleted.
    Additionally there are many one or two word sentences (also called sentence words = words that form a sentence on their own) without any clausal structure like "Okay!" / "Good Night!". 
    Therefore the average sentence length is sometimes shorter than the average clause length, because there are more sentences than clauses, which is highly counter-intuitive at first.
    """
    #number_of_sentences = xml_tree.xpath("count (//s)")
    sent_dict = {}
    for sentence in xml_tree.findall("//s"):
        sent_dict[sentence] = 0
        for word in sentence.findall("w"):
            if word.get("xpos") == "PUNCT" or word.get("xpos") == "SYM":
                continue
            else:
                sent_dict[sentence] += 1
    number_of_sentences = 0
    for key, value in sent_dict.items():
        if value > 0:
            number_of_sentences += 1

    #print(number_of_sentences)
    return int(number_of_sentences)


def count_passives(xml_tree):
    """
    Passives are marked with Voice=Pass in the @feats attribute of a w-element. Because the Voice-part is often
    at the end of the @feats attribute a regular expression is needed to extract the number of passives. Therefore
    the xml_tree gets converted to a string and is then searched with a RegEx.
    """
    xml_tree_as_str = lxml.etree.tostring(xml_tree, encoding="unicode", method="xml")
    passives = re.findall("Voice=Pass", xml_tree_as_str)
    #print(len(passives))
    return int(len(passives))

def count_noun_phrases_and_complex_nominals(xml_tree):
    np_length = 0
    number_of_complex_nominals = 0
    number_of_noun_phrases = 0
    #Find all nouns and their IDs
    noun_ids = xml_tree.xpath("//w[@xpos='NOUN']/@id")
    #Find all heads of all words without including punctuations, probably very slow solution
    heads_for_nps = xml_tree.xpath("//w[not(@xpos='PUNCT')]/@head")
    #A noun phrase can consist of a determiner and a noun. For a complex nominal additional dependents are needed
    heads_for_complex_nominals = xml_tree.xpath("//w[not(@xpos='DET')]/@head")
    
    for noun_id in noun_ids:
        if noun_id in heads_for_nps:
            #If there are words that have a noun as a head it counts as a noun phrase
            number_of_noun_phrases += 1
            #Some nouns have more than one dependent: Here we count how many dependents a noun has and add 1 to include the noun itself in the calculation
            np_length = np_length + heads_for_nps.count(noun_id) + 1
            #If the noun phrase consists of more than just a determiner and a noun, it's counted as a complex nominal: noun+adj, noun+preposition etc.
            if noun_id in heads_for_complex_nominals:
                number_of_complex_nominals += 1
    #Calculate ML of a noun phrase
    mean_length_of_noun_phrases = np_length / number_of_noun_phrases

    return number_of_noun_phrases, number_of_complex_nominals, mean_length_of_noun_phrases

def count_clauses(xml_tree, wordcount):
    """
    Several types of clauses are counted and added together.
    A distinction is made between independent and dependent clauses.
    NOTICE: Due to the nature of subtitles some sentences are very short and don't contain a clause. Example: "Later."
    That leads to an observation that the mean length of a clause can be longer than a sentence. 
    """
    #Dependent adjectival clauses contain relative clauses
    main_clauses = count_main_clauses(xml_tree)
    independent_adjectival_clauses, dependent_adjectival_clauses = count_acl(xml_tree)
    adjectival_clauses = independent_adjectival_clauses + dependent_adjectival_clauses
    adverbial_clauses = count_advcl(xml_tree)
    clausal_complements = count_ccomp(xml_tree)
    open_clausal_complements = count_xcomp(xml_tree)
    parataxis = count_parataxis(xml_tree)
    coordinated_clauses = count_coordinated_clauses(xml_tree)
    clausal_subjects = count_csubj(xml_tree)
    
    #Add all different clauses together depending on their dependability
    independent_clauses = main_clauses + independent_adjectival_clauses + parataxis + coordinated_clauses + clausal_subjects
    dependent_clauses = dependent_adjectival_clauses + clausal_complements + open_clausal_complements + adverbial_clauses
    #Calculate adjectival clauses per 1000 words
    adjectival_clauses_per_1000_words = (adjectival_clauses * 1000) / wordcount
    #Sum up all clauses
    total_clauses = independent_clauses + dependent_clauses
    #print(independent_clauses, dependent_clauses, total_clauses)
    return total_clauses, dependent_clauses, adjectival_clauses_per_1000_words

def count_main_clauses(xml_tree):
    """
    This function looks for sentence heads that are verbs. The definition of a clause is: finite verb + subject
    The search string doesn't look for finiteness, but sometimes the finiteness is marked at the AUX-part of the root.
    """
    independent_clauses = xml_tree.xpath("count (//w[@xpos='VERB' and @deprel='root'])")
    #print(independent_clauses)
    return int(independent_clauses)

def count_acl (xml_tree):
    """
    Counting adjectival clauses. Can be further subdivided into relative clauses.
    Relative clauses are always dependent, regular adjectival clauses are only dependent if they are the head of a subordinating conjunction.
    """
    adjectival_clauses = xml_tree.xpath("count (//w[@deprel='acl'])")
    relative_clauses = xml_tree.xpath("count (//w[@deprel='acl:relcl'])")
    acl_ids = xml_tree.xpath("//w[@deprel='acl']/@id")
    subordinating_conjunction_heads = xml_tree.xpath("//w[@xpos='SCONJ']/@head")
    dependent_acl = 0
    #If a adjectival clause is the head of a subordinating clause, then the ACL is dependent!
    for ID in acl_ids:
        if ID in subordinating_conjunction_heads:
            dependent_acl += 1
    #print(dependent_acl)
    #print(acl_ids, subordinating_conjunction_heads)
    return int(adjectival_clauses - dependent_acl), int(dependent_acl + relative_clauses)

def count_advcl(xml_tree):
    """
    Adverbial clauses are always dependent
    """
    adverbial_clauses = xml_tree.xpath("count (//w[@deprel='advcl'])")
    return int(adverbial_clauses)

def count_csubj(xml_tree):
    """
    Csubj denotes the clausal syntactic subject of a clause, which is itself a clause.
    """
    clausal_subjects = xml_tree.xpath("count (//w[@deprel='csubj'])")
    return int(clausal_subjects)

def count_coordinated_clauses(xml_tree):
    """
    Conjunctions are used to connect sentences. In the UD Tagset independent clauses are connected
    with conjunctions over verbs. So if nouns are connected with conj it's a listing like: An apple, a pear, an orange
    So only verbs with a conj relation are counted: He went to the mall and he bought clothes.
    """
    coordinated_clauses = xml_tree.xpath("count (//w[@xpos='VERB' and @deprel='conj'])")
    #print(coordinated_clauses)
    return int(coordinated_clauses)

def count_ccomp(xml_tree):
    """
    Clausal complements are always dependent
    """
    clausal_complements = xml_tree.xpath("count (//w[@deprel='ccomp'])")
    return int(clausal_complements)

def count_xcomp(xml_tree):
    """
    Open clausal complements. 
    These clauses often don't have an overt subject!
    Sometimes xcomp is a predicative complement, not a clause. 
    Therefore only verbs with xcomp are counted! Parsing is sometimes wrong, as future forms are parsed as present.
    """
    open_clausal_complements = xml_tree.xpath("count (//w[@xpos='VERB' and @deprel='xcomp'])")
    #print(open_clausal_complements)
    return int(open_clausal_complements)

def count_parataxis(xml_tree):
    """
    Parataxis in the UD Tagset can mean a lot of things. If a verb is tagged with parataxis it's mostly a
    coordinated independent clause, that could stand on its own.
    Parataxis is often used for interjections as well and sometimes due to the nature of spoken language
    a whole sentence is marked as parataxis. Nevertheless those parataxis are always independent clauses!
    """
    parataxis = xml_tree.xpath("count (//w[@xpos='VERB' and @deprel='parataxis'])")
    return int(parataxis)

def save_data(syn_features, SYN_CSV):
    syn_df = pd.DataFrame.from_dict(syn_features, orient="index")
    #print(syn_df)
    pd.DataFrame.to_csv(syn_df, SYN_CSV, index_label="IMDB-ID")

def main(FILE_DIR, SYN_CSV):
    syn_features = get_features(FILE_DIR)
    #print(syn_features)
    save_data(syn_features, SYN_CSV)
main(FILE_DIR, SYN_CSV)