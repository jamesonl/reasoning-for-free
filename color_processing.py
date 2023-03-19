
from collections import namedtuple

import matplotlib.patches as mpatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color as color
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split

import utils
from utils import END_SYMBOL, START_SYMBOL, UNK_SYMBOL

utils.fix_random_seeds()

from nltk.tokenize import word_tokenize

rep = dict()
rep[""]=""
rep["brite"]="bright"
rep["lavendar"]="lavender"
rep["yello"]="yellow"
rep["brighte"]="bright"
rep["brigh"]="bright"
rep["florescent"]="fluorescent"
rep["pupleish"]="purplish"
rep["avacado"]="avocado"
rep["tanish"]="tannish"
rep["browinsh"]="brownish"
rep["greylish"]="greyish"
rep["briht"]="bright"
rep["drker"]="darker"
rep["gren"]="green"
rep["drk"]="dark"
rep["neonish"]="neonish"
rep["greeny"]="green"
rep["blueer"]="bluer"
rep["gry"]="grey"
rep["camoflage"]="camouflage"
rep["bbright"]="bright"
rep["puple"]="purple"
rep["pruple"]="purple"
rep["greensih"]="greenish"
rep["greeen"]="green"
rep["puirple"]="purple"
rep["fuschia"]="fuchsia"
rep["grnish"]="greenish"
rep["puplish"]="purplish"
rep["dul"]="dull"
rep["darke"]="dark"
rep["turqoise"]="turquoise"
rep["bue"]="blue"
rep["purpler"]="purple"
rep["s"]=""
rep["blu"]="blue"
rep["gree"]="green"
rep["drabest"]="drabbest"
rep["brihgt"]="bright"
rep["greyist"]="greyish"
rep["purpe"]="purple"
rep["purply"]="purple"
rep["mustardy"]="mustard"
rep["puprle"]="purple"
rep["purpleish"]="purplish"
rep["dk"]="dark"
rep["browish"]="brownish"
rep["greyis"]="greyish"
rep["greysih"]="greyish"
rep["purplr"]="purple"
rep["brightish"]="brightish"
rep["purpley"]="purplish"
rep["bluw"]="blue"
rep["yellwo"]="yellow"
rep["inbetween"]="between"
rep["blueish"]="bluish"
rep["purp"]="purple"
rep["greenn"]="green"
rep["purpil"]="purple"
rep["purpl"]="purple"
rep["flourescent"]="fluorescent"
rep["palish"]="pailish"
rep["urple"]="purple"
rep["lighest"]="lightest"
rep["prurple"]="purple"
rep["brownis"]="brownish"
rep["pinl"]="pink"
rep["grn"]="green"
rep["brigher"]="brighter"
rep["bight"]="bright"
rep["lueish"]="blueish"
rep["brighest"]="brightest"
rep["bronw"]="brown"
rep["pinkl"]="pink"
rep["blueest"]="bluest"
rep["lizzard"]="lizard"
rep["greeeen"]="green"
rep["fuscia"]="fascia"
rep["purpple"]="purple"
rep["blye"]="blue"
rep["orangeish"]="oranges"
rep["brwnish"]="brownish"
rep["marroon"]="maroon"
rep["ornage"]="orange"
rep["grreen"]="green"
rep["brwon"]="brown"
rep["turqouise"]="turquoise"
rep["pnk"]="pink"
rep["orangy"]="orange"
rep["yelow"]="yellow"
rep["blyue"]="blue"
rep["tourqoise"]="turquoise"
rep["periwickle"]="periwinkle"
rep["purpkle"]="purple"
rep["bllue"]="blue"
rep["prpl"]="purple"
rep["pueple"]="purple"
rep["greyets"]="greyest"
rep["redist"]="reddest"
rep["turquise"]="turquoise"
rep["ppurple"]="purple"
rep["turquiose"]="turquoise"
rep["kaki"]="khaki"
rep["turquose"]="turquoise"
rep["greeb"]="green"
rep["yel"]="yellow"
rep["pirple"]="purple"
rep["oragne"]="orange"
rep["greenist"]="greenest"
rep["yellowy"]="yellow"
rep["camoflauge"]="camouflage"
rep["bronw"]="brown"
rep["kacky"]="kahki"
rep["piink"]="pink"
rep["yelowish"]="yellowish"
rep["purle"]="purple"
rep["yelllow"]="yellow"
rep["pinkl"]="pink"
rep["sagey"]="sage"
rep["blueest"]="bluest"
rep["bluey"]="blue"
rep["daker"]="darker"
rep["seafoam"]="seafood"
rep["tourquoise"]="turquoise"
rep["greem"]="green"
rep["lizzard"]="lizard"
rep["purpel"]="purple"
rep["pinkest"]="finest"
rep["almosy"]="almost"
rep["brigtest"]="brightest"
rep["greeeen"]="green"
rep["fuscia"]="fascia"
rep["voilet"]="violet"
rep["grene"]="green"
rep["oranage"]="orange"
rep["jeeze"]="jeez"
rep["rd"]="red"
rep["fusia"]="fuschia"
rep["rpurple"]="purple"
rep["hte"]="the"
rep["lemony"]="lemon"
rep["purpple"]="purple"
rep["yellower"]="yellowed"
rep["blye"]="blue"
rep["ehh"]="heh"
rep["brwnish"]="brownish"
rep["birght"]="bright"
rep["marroon"]="maroon"
rep["ornage"]="orange"
rep["grreen"]="green"
rep["brwon"]="brown"
rep["coloe"]="color"
rep["grenish"]="greenish"
rep["turqouise"]="turquoise"
rep["fushia"]="fuschia"
rep["bluesih"]="blueish"
rep["ligther"]="lighter"
rep["ligher"]="lighter"
rep["borwn"]="brown"
rep["piurple"]="purple"
rep["pnk"]="pink"
rep["greish"]="greyish"
rep["mauvey"]="mauve"
rep["roseish"]="roses"
rep["orangy"]="orange"
rep["yellowist"]="yellowest"
rep["aquaish"]="squash"
rep["grren"]="green"
rep["collor"]="color"
rep["rosey"]="rose"
rep["lolol"]="lol"
rep["viloet"]="violet"
rep["olivey"]="olive"
rep["blueist"]="bluest"
rep["redest"]="reddest"
rep["brught"]="bright"
rep["lightes"]="lightest"
rep["purplelish"]="purplish"
rep["mauveish"]="mauve"
rep["yelow"]="yellow"
rep["grrey"]="grey"
rep["mauve"]="mauve"
rep["yelloy"]="yellow"
rep["ooops"]="oops"
rep["oilve"]="olive"
rep["pinkier"]="pinkie"
rep["blyue"]="blue"
rep["tourqoise"]="tortoise"
rep["tradional"]="traditional"
rep["redish"]="reddish"
rep["sherbert"]="sherbet"
rep["oceanlike"]="ocean"
rep["periwickle"]="periwinkle"
rep["graish"]="greyish"
rep["auqa"]="aqua"
rep["purpkle"]="purple"
rep["brn"]="brown"
rep["clostest"]="closest"
rep["bllue"]="blue"
rep["slighly"]="slightly"
rep["salmony"]="salmon"
rep["prpl"]="purple"
rep["plue"]="blue"
rep["pueple"]="purple"
rep["greyets"]="greyest"
rep["blueish"]="bluish"
rep["brughter"]="brighter"
rep["perriwinkle"]="periwinkle"
rep["irange"]="range"
rep["birhgt"]="bright"
rep["bownish"]="brownish"
rep["redist"]="reddist"
rep["turquise"]="turquoise"
rep["ylow"]="yellow"
rep["ummmm"]="ummm"
rep["ppurple"]="purple"
rep["fuesha"]="fuschia"
rep["turquiose"]="turquoise"
rep["kaki"]="khaki"
rep["theyre"]="there"
rep["dullish"]="bullish"
rep["turquose"]="turquoise"
rep["cloest"]="closest"
rep["greeb"]="green"
rep["pirple"]="purple"
rep["yel"]="yellow"
rep["prple"]="purple"
rep["purplely"]="purple"
rep["teals"]="teal"
rep["slighty"]="slightly"
rep["oragne"]="orange"
rep["browny"]="brown"
rep["yeloow"]="yellow"
rep["greenist"]="greenest"
rep["oastel"]="pastel"
rep["yellowy"]="yellow"
rep["purlpl"]="purple"
rep["purpl"]="purple"
rep["brigth"]="bright"
rep["brow"]="brown"
rep["redd"]="red"
rep["reddist"]="reddest"
rep["purpul"]="purple"


def respell(s):
    if s in rep.keys() :
        s = rep[s]

    return s

def tokenize_example(s):

    s = s.lower().replace("gray","grey")
    s = s.replace("/"," ")
    s = s.replace("-"," ")

    words = word_tokenize(s)
    words = ["".join(ch for ch in word if ch.isalnum()) for word in words]
    words = [respell(word) for word in words if (len(word) > 1 and word.isnumeric() == False) ]
    s     = " ".join(words) + " "
    # s     = s.replace("er "," er ")
    # s     = s.replace("est "," est ")
    # s     = s.replace("ish "," ish ")
    # s     = s.replace("ly "," ly ")
    words = [word for word in s.split(" ") if word != ""] 
    words = [respell(word) for word in words]   

    return words 


def create_embeddings(colors_list, text_list, color_rep) :
    dim_color_reps = np.concatenate(color_rep( colors_list[0] )).shape[0] 
    sentences = [  tokenize_example( sentence )  for sentence in text_list  ]
    vocab     = utils.get_vocab(sentences,mincount=2)
    identity  = np.identity(len(vocab)+dim_color_reps,dtype=np.float32)
    embedding = { word:identity[k]for k,word in enumerate(vocab) }
    color_reps = [ color_rep( colors ) for colors in colors_list ]
    embeddings = [ [ embedding[word] for word in sentence ] for sentence in sentences]
    return color_reps, embeddings



def display_colors(rgb_colors,text):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(3,1))
    for i, c in enumerate(rgb_colors):
        patch = mpatch.Rectangle((0, 0), 1, 1, color=c, ec="white", lw=8)
        axes[0][i].add_patch(patch)
        axes[0][i].axis('off')
        axes[1][i].axis('off')
    axes[1][1].annotate(text, (0, 0), color='b', weight='bold', fontsize=10, ha='center', va='center')

def convert_to_lab(rgb_colors):
    nprgb = np.array(rgb_colors)
    return color.rgb2lab(nprgb) 





