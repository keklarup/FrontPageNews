# script for pulling stories out of PDF copies of front pages.
# Broadly, script has __ sections:

#importing needed modules:

import time
import pandas as pd #to write/read/work with csv files.
import numpy as np #to do math
#import seaborn as sns #for additional plot features
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import matplotlib.cm as cm

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
# From PDFInterpreter import both PDFResourceManager and PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
# Import this to raise exception whenever text extraction from PDF is not allowed
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from pdfminer.converter import PDFPageAggregator

####################
###
### Definitions
###
####################

def extract_text(my_file):
    """Pulling text boxes out of PDFs. First half of this defn copies off the internet."""
    try:
        #my_file = os.path.join(base_path + "/" + filename)
        #my_file = os.path.join(dayDataPath, frontPages[paper])
        password = ""
        extracted_text = ""
        extracted_text_plus=[];
        # Open and read the pdf file in binary mode
        fp = open(my_file, "rb")
        # Create parser object to parse the pdf content
        parser = PDFParser(fp)
        # Store the parsed content in PDFDocument object
        document = PDFDocument(parser, password)
        # Check if document is extractable, if not abort
        #if not document.is_extractable:
        #    raise PDFTextExtractionNotAllowed
        # Create PDFResourceManager object that stores shared resources such as fonts or images
        rsrcmgr = PDFResourceManager()
        # set parameters for analysis
        laparams = LAParams()
        # Create a PDFDevice object which translates interpreted information into desired format
        # Device needs to be connected to resource manager to store shared resources
        # device = PDFDevice(rsrcmgr)
        # Extract the decive to page aggregator to get LT object elements
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        # Create interpreter object to process page content from PDFDocument
        # Interpreter needs to be connected to resource manager for shared resources and device 
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Ok now that we have everything to process a pdf document, lets process it page by page
        for page in PDFPage.create_pages(document):
            # As the interpreter processes the page stored in PDFDocument object
            interpreter.process_page(page)
            # The device renders the layout from interpreter
            layout = device.get_result()
            # Out of the many LT objects within layout, we are interested in LTTextBox and LTTextLine
            for lt_obj in layout:
                #print(lt_obj)
                #extracted_text_plus.append(lt_obj)
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    extracted_text_plus.append(lt_obj)
            #print(layout)
        #close the pdf file
        fp.close()
        #save the text
        #with open(log_file, "wb") as my_log:
        #    my_log.write(extracted_text.encode("utf-8"))
        
    ###Finally getting to my contributions.###
    #Headlines are assumed to be large text. By comparing the number of lines of text in a textbox
    #with the height of the textbox, the average size of the text can be found.
    #Text that's larger than average is kept.
        df=pd.DataFrame()
        df['cords']=0
        df['num']=0;
        df['height']=0;
        df['text']=''
        df['TL_X']=-1
        df['TL_Y']=-1
        df['width']=-1
        nums=[];
        heights=[];
        for n in range(0,len(extracted_text_plus)):
            cords=str(extracted_text_plus[n]).split(' ')[1].split(',')
            vals=[float(elm) for elm in cords]
            a,b,c,d=vals
            text=' '.join(str(extracted_text_plus[n]).split(' ')[2:])
            h=d-b#float(cords[3])-float(cords[1])
            w=c-a
            #nums.append(n)
            #heights.append(h)
            #print(cords)
            df.loc[n,'cords']=' '.join(cords)
            df.loc[n,'num']=n
            df.loc[n,'height']=h
            df.loc[n,'width']=w
            df.loc[n,'TL_X']=a
            df.loc[n,'TL_Y']=b
            df.loc[n,'text']=text
        df['newlines']=0;
        for x in range(0,len(df)):
            df.loc[x,'newlines']=df.loc[x,'text'].count('\\n')
        df['text height']=df['height']/df['newlines']
        return df
    except:
        pass

def visPaper(fig, ax, df, color='r', figureWidth=10, figureHeight=10, fill=False):
    """For visualizing the extracted text boxes"""
    ax.plot()
    ax.set_xlim([0,figureWidth])
    ax.set_ylim([0,figureHeight])
    if fill==True:
        bodyColor=color
    else:
        bodyColor='none'
    for vals in df.loc[:,['TL_X','TL_Y','width','height']].values:
        #vals=[float(elm) for elm in vals]
        a,b,c,d=vals
        rect = patches.Rectangle((a,b),c,d,linewidth=1,edgecolor=color,facecolor=bodyColor)
        ax.add_patch(rect)    

def identifyHeadlines(df, makeImage=False):
    paperWidth=(df['TL_X']+df['width']).max() #for figures
    paperHeight=(df['TL_Y']+df['height']).max() #for figures
    #identify headlines:
    df['headline']=False
    df.loc[df['text height']>df['text height'].mean()+0.5*df['text height'].std(),'headline']=True
    #giving each headline a group
    df['group']=-1
    x=0
    for i in df.loc[df.loc[:,'headline']==True,:].index:
        df.loc[i,'group']=x
        x=x+1
    #combine split headlines:
    headlineIndices=df.loc[df.loc[:,'headline']==True,:].index;
    for x in range(0, len(headlineIndices)):
        verLims=(df.loc[headlineIndices,'TL_Y']<(df.loc[headlineIndices[x],'TL_Y']+2*df.loc[headlineIndices[x],'height'])) &\
            (df.loc[headlineIndices,'TL_Y']>df.loc[headlineIndices[x],'TL_Y'])
        horLims=np.abs((df.loc[headlineIndices,'TL_X']+(df.loc[headlineIndices,'TL_X']+df.loc[headlineIndices,'width'])/2)-\
            (df.loc[headlineIndices[x],'TL_X']+(df.loc[headlineIndices[x],'TL_X']+df.loc[headlineIndices[x],'width'])/2))<\
            df.loc[headlineIndices[x],'width']*.2
        merge=headlineIndices[verLims&horLims]
        if len(merge)>0:
            df.loc[headlineIndices[x],'group']=df.loc[merge[0],'group']
    if makeImage==True:
        colors=np.array([[.01, .01, .001, 1]]+list(cm.rainbow(np.linspace(0,1,len(df['group'].unique())))))
        c=0
        fig,ax = plt.subplots(1,figsize=(4,6))
        for group in sorted([elm for elm in df.loc[:,'group'].unique()]):
            visPaper(fig, ax, df.loc[df.loc[:,'group']==group,:], color=colors[c],
                    figureWidth=paperWidth, figureHeight=paperHeight)
            c=c+1
        plt.show()
    return df, paperWidth, paperHeight

def linkText(df, makeImage=False):
    """Given headlines, use some hard coded solutions to associate body text with headlines."""
    df['center']=df['TL_X']+.5*df['width']
    paperWidth=(df['TL_X']+df['width']).max() #for figures
    paperHeight=(df['TL_Y']+df['height']).max() #for figures
    for group in [elm for elm in df.loc[:,'group'].unique() if elm >=0]:
        #print(group)
        #A,B,C,D=df.loc[(df.loc[:,'headline']==True)&(df.loc[:,'group']==group),['TL_X','TL_Y','width','height']].values[0]
        A,B,C,D=df.loc[(df.loc[:,'headline']==True)&(df.loc[:,'group']==group),['TL_X','TL_Y','width','height']]\
        .sort_values(by='width',ascending=False).values[0]
        textPossibles=df.loc[(df.loc[:,'TL_Y']<B)&(df.loc[:,'center']>A)&(df.loc[:,'center']<=A+C),:].copy()
        #print(textPossibles.index)
        targetHeadline=df.loc[df.loc[:,'group']==group,:]
        Hindex=targetHeadline.sort_values(by='width',ascending=False).index[0]
        #print(Hindex)
        targetHeadline=df.loc[Hindex,:]
        try:
            textCutOff=df.loc[(df.loc[:,'group']!=group)&\
                        (df.loc[:,'headline']==True)&(df.loc[:,'TL_Y']<targetHeadline.loc['TL_Y'])&\
                        (((df.loc[:,'TL_X']+df.loc[:,'width'])>targetHeadline.loc['TL_X'])|\
                       (df.loc[:,'TL_X']<targetHeadline.loc['TL_X']+targetHeadline.loc['width'])),:]\
            .sort_values(by=['TL_Y']).index[0]
            textIndices=textPossibles[textPossibles['TL_Y']>textPossibles.loc[int(textCutOff),'TL_Y']].index
        except:
            textIndices=textPossibles.index
        textIndices=[elm for elm in textIndices if df.loc[elm,'headline']==False]
        df.loc[textIndices,'group']=group 
    if makeImage==True:
        colors=np.array([[.01, .01, .001, 1]]+list(cm.rainbow(np.linspace(0,1,len(df['group'].unique())))))
        c=0
        fig,ax = plt.subplots(1,figsize=(4,6))
        for group in sorted([elm for elm in df.loc[:,'group'].unique()]):
            visPaper(fig, ax, df.loc[df.loc[:,'group']==group,:], color=colors[c], 
                     figureWidth=paperWidth, figureHeight=paperHeight, fill=False)
            c=c+1
        colors=np.array([[.01, .01, .001, 1]]+list(cm.rainbow(np.linspace(0,1,len(df['group'].unique())))))
        c=0
        for group in sorted([elm for elm in df.loc[:,'group'].unique()]):
            visPaper(fig, ax, df.loc[(df.loc[:,'group']==group)&(df.loc[:,'headline']==True),:], color=colors[c], 
                     figureWidth=paperWidth, figureHeight=paperHeight, fill=True)
            c=c+1
        plt.show()
    return df

def cleanText(text):
    return text.replace("'",' ').replace('\\n',' ').replace('>',' ').replace('\\x',' ').replace('- ','')

def sortText(df, group):
    textIndices=df.loc[(df.loc[:,'group']==group)&(df.loc[:,'headline']==False),:].index
    textPossiblesSorted=df.loc[textIndices,:]
    textPossiblesSorted['center']=textPossiblesSorted['TL_X']+.5*textPossiblesSorted['width']
    textPossiblesSorted=textPossiblesSorted.sort_values(by='center').copy()
    textPossiblesSorted['column']=-1
    i=0
    while -1 in textPossiblesSorted['column'].values:
        testVal=textPossiblesSorted.loc[textPossiblesSorted.loc[:,'column']==-1,:].index[0]
        test=textPossiblesSorted.loc[testVal, 'center']
        textPossiblesSorted.loc[np.abs(textPossiblesSorted['center']-test)<10,'column']=i
        i=i+1
    textPossiblesSorted=textPossiblesSorted.sort_values(by='TL_Y',ascending=False).sort_values(by='column')
    textIndicesPossible=textPossiblesSorted.index;
    textIndices=[]
    for i in textIndicesPossible:
        if cleanText(df.loc[i,'text']).strip().isupper()!=True:
            textIndices.append(i)
    return textIndices

def extractedStories(df, filename=''):
    groups=[elm for elm in df['group'].unique() if elm>=0]
    df_extracted=pd.DataFrame(index=range(len(groups)), columns=['filename','headline','body'])
    df_extracted['filename']=filename
    i=0
    for group in groups:
        Headline=cleanText(' '.join(df.loc[(df.loc[:,'group']==group)&(df.loc[:,'headline']==True),'text'].values))
        #bodyText=cleanText(' '.join(df.loc[(df.loc[:,'group']==group)&(df.loc[:,'headline']==False),'text'].values))
        bodyText=cleanText(' '.join(df.loc[sortText(df, group),'text'].values))
        df_extracted.loc[i,'headline']=Headline;
        df_extracted.loc[i,'body']=bodyText
        i=i+1
    return df_extracted

US_labels=['CA', 'PA', 'FL', 'OH', 'DE', 'TX', 'MA', 'NC', 'IN', 'NY', 'CT', 'VA', 'MI',
       'SC', 'WI', 'CO', 'GA', 'LA', 'WA', 'NJ', 'KY', 'MN', 'TN', 'AL', 'IL',
       'ID', 'WV', 'ME', 'VT', 'ND', 'SD', 'NM', 'MD', 'IA', 'MO', 'AR', 'HI',
       'AK', 'MT', 'OK', 'AZ', 'MS', 'KS', 'NE', 'NV', 'OR', 'UT', 'RI', 'DC',
       'WY', 'WSJ.pdf', 'NH']

##############
###
### Script portion
###
##############

print('This script assumes PDFs of front pages have already been scraped from Newseum using webscrapeNewseum.py.')
print('For which previously scraped day should this script extract stories (yyyymmdd)?')
day=str(input())
dayDataPath='data/%s'%(day)
frontPages=os.listdir(dayDataPath)
print('Starting with %s pages.'%(len(frontPages)))
UsaFrontPages=[elm for elm in frontPages if elm.split('_')[1] in US_labels]
print('Looking only at USA papers, so population down to', len(UsaFrontPages))

# Option to look at how headline extraction and body text association pieces work.
print('Visualize some examples before running at scale (y/n)?')
userInput=input().lower()
while userInput=='y':
    check2=0
    while check2==0:
        print('Either provide paper code (probably similar to "DC_WP") or just state abbreviation to see list of options.')
        userInput2=input().upper()
        if len(userInput2)==2:
            print('Possibles:')
            print([elm for elm in UsaFrontPages if userInput2 in elm.split('_')[1]])
        else:
            try:
                target=[elm for elm in UsaFrontPages if userInput2 in elm][0]
                paper=list(UsaFrontPages).index(target)
                my_file = os.path.join(dayDataPath, UsaFrontPages[paper])
                df=extract_text(my_file)
                print(len(df))
                print('data/%s/%s'%(day,UsaFrontPages[paper]))
                df,paperWidth, paperHeight=identifyHeadlines(df, makeImage=False)
                df=linkText(df, makeImage=True)
                df_extracted=extractedStories(df)
                print(df_extracted)
                check2=1
            except:
                print('Failed to find that particular code. Please try again.')
    print('Look at another example (y/n)?')
    userInput=input().lower()
    
print("""
Next step to to attempt extraction from every PDF.
This can take several minutes.
""")

datapath="ExtractedDailyStories"
print("""
Currently hard coded to save extracted data as .csv as %s
"""%(os.path.join(datapath,'extractedPairs_%s.csv'%(day))))
print("""
Proceed (y/n)?
""")
userInput=input().lower()
if userInput == 'y':
    df_extractedComposite=pd.DataFrame(columns=['filename','headline','body'])
    for paper in range(0,len(UsaFrontPages)):
        if paper%25==0:
            print(paper)
        try:
            #filename=UsaFrontPages[paper]
            my_file = os.path.join(dayDataPath, UsaFrontPages[paper])
            df=extract_text(my_file)
            #print(len(df))
            if len(df)>0:
                paperWidth=(df['TL_X']+df['width']).max() #for figures
                paperHeight=(df['TL_Y']+df['height']).max() #for figures
                #print('file:///C:/Users/Kyle/Documents/Projects/WorldNews/data/20190809/%s'%(UsaFrontPages[paper]))
                df,paperWidth, paperHeight=identifyHeadlines(df, makeImage=False)
                df=linkText(df, makeImage=False)
                df_extracted=extractedStories(df,filename=UsaFrontPages[paper])
                df_extractedComposite=pd.concat([df_extractedComposite,df_extracted])
            else:
                print('zero headlines found on ', paper)
        except:
            print('something failed on ', paper)     
    #####Save data:
    df_extractedComposite.to_csv(os.path.join(datapath,'extractedPairs_%s.csv'%(day)), index=False)