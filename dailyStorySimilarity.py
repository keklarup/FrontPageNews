# File for identifying similar stories from a day's collection of stories

import os
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cossim
from sklearn import decomposition
import networkx as nx
import matplotlib.pyplot as plt
import time
import re

############
###
### Definitions
###
############

option=1
try:
    nltk.download('averaged_perceptron_tagger')
except:
    option=2
def POSFreq(sentence, option=option):
    """
    Helps to identify which extracted stories actually have body text and which are noise.
    For example: newspaper name may be extracted as headline with the masthead(?) as only bit of body.
    We want to drop that.
    """
    if option==1:
        POSCounts=nltk.FreqDist(tag for (word, tag) in nltk.pos_tag(nltk.word_tokenize(sentence)))
        if POSCounts['NNP']>20:
            return True
        else:
            return False
    else:
        if len(sentence)>250: #simple check of how long the story is.
            return True
        else:
            return False

def cleanText(text):
    """
    Simple regex to clean up the text slightly.
    """
    return text.lower().replace("'",' ').replace('\\n',' ').replace('>',' ').replace('\\x',' ').replace('- ','')

def cleanData(df):
    print("size before cleaning, the number of stories in day's dataset:", len(df))
    #limit to only U.S. papers:
    df=df.loc[df.loc[:,'filename'].str.split('_').str[1].isin(US_labels),:].copy()
    #remove stories where headline longer than associated body text:
    df=df.loc[df.loc[:,'headline'].str.len()<df.loc[:,'body'].str.len(),:].copy()
    print("After first pass of cleaning, this number of stories left in day's dataset:", len(df))

    #remove stories where body text too small:
    t1=time.time()
    df=df.loc[df.loc[:,'body'].apply(lambda text: POSFreq(text, option=2)),:].copy() #nltk NNP option takes much longer.
    df.reset_index(inplace=True)
    df=df.rename(columns={'index':'originalindex'})
    t2=time.time()
    print("Time taken for second round of cleaning:", t2-t1)
    print("After second pass of cleaning, this number of stories left in day's dataset:", len(df))

    #regex cleaning:
    df.loc[:,'body']=df.loc[:,'body'].apply(lambda text: cleanText(text))
    return df

def graphAll(df, max_features=1000, keep_unconnected=False, lowerThresh=.9, upperThresh=1.1):
    """
    Option for graphing the similarity of stories.
    If keep_unconnected == True, nodes with no connecting edges will be included.
    df -- contains the stories (newspaper name, headlines, body text)
    max_features -- dimension of TFiDF vectorization of body text from collection of day's stories
    lowerThresh -- lower bound for similarity of cossim measure for connecting story nodes
    upperThresh -- upper bound to lowerThresh
    """
    corpus=df.loc[:,'body'].fillna('').str.lower().values
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(corpus)
    G = nx.Graph()
    edges=[]
    for x in range(0,len(df)):
        simMeasures=cossim(X[x],X)
        matches=df.loc[(simMeasures[0]>=lowerThresh)&(simMeasures[0]<=upperThresh),:].index
        if keep_unconnected==False:
            matches=[elm for elm in matches if elm!= x]
        for elm in matches:
            edges.append((x,elm))
    G.add_edges_from(edges)
    return G, vectorizer

def graphAddedEdges(G, df, vectorizer, cosSim_thresh=.5):
    """
    Deals with the issue that same topic was split into a few different clusters thanks to language nuance 
    (and potentially the different amount of space each editor gave the story's body text).
    Links groups with similar average cos sim score.
    """
    connectedStories=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True) 
    connectedStoriesCount=[len(elm.nodes()) for elm in connectedStories]
    #finding subgraphs:
    subGraphs=range(0,connectedStoriesCount.index(2))
    dfSub=pd.DataFrame(columns=['filename','headline','body','GraphGroup'])
    for i in subGraphs:
        dfSub0=df.loc[list(connectedStories[i].nodes()),:]
        dfSub0['GraphGroup']=i
        dfSub=pd.concat([dfSub,dfSub0], sort=True)
    lsa=decomposition.TruncatedSVD(n_components=5, algorithm='randomized',n_iter=5)
    Xsub = vectorizer.transform(dfSub.loc[:,'body'].values)
    XsubLsa=lsa.fit_transform(Xsub)

    #finding mean cos sim score for each subgroup:
    groupMeans=np.array(np.mean(Xsub[(dfSub['GraphGroup']==subGraphs[0]).values],axis=0))
    for i in subGraphs[1:]:
        groupMeans=np.concatenate([groupMeans, np.array(np.mean(Xsub[(dfSub['GraphGroup']==subGraphs[i]).values],axis=0))])
        
    #linking subgroups with rel. high cos sim score
    addedEdges=[]
    combined_groups=[]
    for i in subGraphs:#range(22,len(subGraphs)):#subGraphs:
        simScores=cossim(groupMeans[i].reshape(1,-1),groupMeans)
        possibleMissedconnections=[elm[0] for elm in list(zip(range(0,len(simScores[0])),(simScores>cosSim_thresh)[0])) if elm[1]==True]
        combined_groups.append(possibleMissedconnections)
        if len(possibleMissedconnections)>1:
            for j in [elm for elm in possibleMissedconnections if elm != i]:
                u=list(connectedStories[i].nodes)[0]#np.random.choice(np.array(connectedStories[i].nodes))
                v=list(connectedStories[j].nodes)[0]#np.random.choice(np.array(connectedStories[j].nodes))
                G.add_edge(u,v)
                addedEdges.append((u,v))
    
    # Finding the clusters of larger groups
    metaGrouped=[]
    for i in range(0,len(combined_groups)):
        grouped=[]
        for elm in combined_groups[i]:
            for group in combined_groups[i:]:
                if elm in group:
                    grouped.extend(group)
        metaGrouped.append(sorted(list(set(grouped))))
    newGroups=[]
    for i in range(0,len(metaGrouped)):
        if metaGrouped[i][0]==i:
            newGroups.append(metaGrouped[i])
    return dfSub, newGroups, addedEdges

def largerGroupingCheck(dfSub, newGroups):
    """
    Small def for printing out the common headline n-grams from the original groups that have been combined.
    Decent check of how well I'm doing at getting the same stories together with the second clustering step.
    """
    for Group in range(0, len(newGroups)):
        print('\n New Group:',Group)
        for g in newGroups[Group]:
            try:
                tfidf=TfidfVectorizer(stop_words='english', max_features=5)
                corpus=dfSub.loc[dfSub.loc[:,'GraphGroup']==g,'headline'].str.strip().str.lower().values
                tfidf.fit(corpus)
                topic='|'.join(tfidf.get_feature_names())
                print(topic, '||| (Num. Stories:', len(corpus), ')')
            except:
                print('Fail. Probably on fitting headline corpus')

def plotSubgroup(G, df, groupNum=0, subGraphVis=True):
    connectedStories=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    Gc=connectedStories[groupNum]
    try:
        #print(df.loc[sorted(list(Gc.nodes)),'headline'].sample(5))
        tfidf=TfidfVectorizer(stop_words='english', max_features=5)
        corpus=df.loc[sorted(list(Gc.nodes)),'headline'].str.strip().str.lower().values
        tfidf.fit(corpus)
        topic='|'.join(tfidf.get_feature_names())
        print(topic, len(corpus))
    except:
        print(df.loc[sorted(list(Gc.nodes)),'headline'])
    print('number of nodes:',len(list(Gc.nodes)))
    if subGraphVis==True:
        fig,ax = plt.subplots(1,figsize=(12,6))
        nx.draw(Gc,with_labels=True,node_size=600, node_color='lightgreen')
        plt.title(topic)
        plt.show()
    return Gc, topic
  
    
#removing bridge nodes that have a chance to link 2 different clusters:
def bridgeNodeRemoval(G):
    connectedStories=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
    cullNodes=[]
    for group in connectedStories[0:10]:
        Gc0=group
        oldCount=nx.number_connected_components(Gc0)
        for node in list(Gc0.nodes):
            Gct=Gc0.copy()
            Gct.remove_node(node)
            newCount=nx.number_connected_components(Gct)
            if newCount>oldCount:
                cullNodes.append(node)
    #G2=G.copy()
    for node in cullNodes:
        G.remove_node(node)
    return G, cullNodes

#below: US_labels used to drop world newspapers and only focus on U.S. papers.
US_labels=['CA', 'PA', 'FL', 'OH', 'DE', 'TX', 'MA', 'NC', 'IN', 'NY', 'CT', 'VA', 'MI',
       'SC', 'WI', 'CO', 'GA', 'LA', 'WA', 'NJ', 'KY', 'MN', 'TN', 'AL', 'IL',
       'ID', 'WV', 'ME', 'VT', 'ND', 'SD', 'NM', 'MD', 'IA', 'MO', 'AR', 'HI',
       'AK', 'MT', 'OK', 'AZ', 'MS', 'KS', 'NE', 'NV', 'OR', 'UT', 'RI', 'DC',
       'WY', 'WSJ.pdf', 'NH']


def workDay(day, save=False, fullGraphVis=True, histVis=True, subGraphVis=True, subGraphs=[0,1,2]):
    """Main Defn of this script. Generates range of visuals for a given day.
       Returns dataframe (df) and graph (G)"""
    if len(day)==8:
        day='extractedPairs_%s.csv'%(day)
    print(day)
    datapath="ExtractedDailyStories"
    #df=pd.read_csv(os.path.join(datapath,'extractedPairs_%s.csv'%(day)))
    df=pd.read_csv(os.path.join(datapath,day))
    #clean data:
    df=cleanData(df)
    #df.loc[262,'body']=''
    print("Number of unique paper's in day's data:", len(df.loc[:,'filename'].unique()))
    #vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    #make graph:
    G, vectorizer=graphAll(df, keep_unconnected=False, lowerThresh=.6, upperThresh=1.1)

    #####added bit######
    # remove bridge nodes
    G, cullNodes=bridgeNodeRemoval(G)
    #####reg. scripts###

    #find similar groups to link:
    dfSub, newGroups, addedEdges=graphAddedEdges(G, df, vectorizer,cosSim_thresh=.5)
    #check groupings:
    #largerGroupingCheck(dfSub, newGroups)
    #make graph:
    if fullGraphVis==True:
        fig,ax = plt.subplots(1,figsize=(18,12))
        nx.draw(G, with_labels=False,node_size=2, node_color='lightgreen', 
                edge_color=['red' if e in addedEdges else 'black' for e in G.edges])
        plt.show()
    # histogram of group sizes
    if histVis==True:
        connectedStories=sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)
        connectedStoriesCount=[len(elm.nodes()) for elm in connectedStories]
        plt.bar(x=range(len(connectedStoriesCount)), height=connectedStoriesCount);
        plt.ylabel('Number of similar stories')
        plt.xlabel('Group Number')
        plt.title("Histogram of group sizes for %s"%(day));
        plt.show()
    #looking at particular subgroups in detail:
    for n in subGraphs:
        Gcn, topic=plotSubgroup(G, df, groupNum=n, subGraphVis=subGraphVis)
        print('Topic for subGraph:', topic)
    #saving:
    if save==True:
        savepath="GroupedDailyStories"
        df['group']=-1
        for i in range(0,len(connectedStories)):
            df.loc[list(connectedStories[i].nodes()),'group']=i
        #merge with original data
        df2=pd.read_csv(os.path.join(datapath,day))
        df2.reset_index(inplace=True)
        df2=df2.rename(columns={'index':'originalindex'})
        df2=df2.merge(df[['filename','originalindex','group']], how='left',on=['filename','originalindex']).copy()
        print('groupedStories_%s'%(day.split('_')[1]))
        df2.to_csv(os.path.join(savepath,'groupedStories_%s'%(day.split('_')[1])))
    return df, G

###############
###
### Script portion
###
###############

print("""
Script for identifying groupings of similar stories in a day's worth of news.

This script expects there to already exist a csv with extracted stories from the PDFs.
Currently, that csv name is hard coded as 'extractedPairs_yyyymmdd.csv'.
This csv is generated with the extractStories.py file.
""")

cont=2
while cont>1:
    print("""
    Please provide one of the following:
       1) A single day for which to provide groupings (format: yyyymmdd),
       2) A range of days. Grouping analysis will run on each day individually (format: yyyymmdd-yyyymmdd),
       3) A keyword from the following options:
            * 'records' -- list all available days with csv records
            * 'end' -- stop program
    """)


    datapath="ExtractedDailyStories"
    dataDays=os.listdir(datapath)
    userInput=str(input()).lower()
    if len(userInput)==8:
        print('Running on single day')
        try:
            day=userInput
            dataDays=[[elm for elm in dataDays if userInput in elm][0]]
            cont=1
        except:
            print('Record csv not found.')
    elif len(userInput.split('-'))==2 and len(userInput.split('-')[0])==8 and len(userInput.split('-')[1])==8:
        print('Will run across span of days.')
        start=dataDays.index([elm for elm in dataDays if userInput.split('-')[0] in elm][0])
        stop=dataDays.index([elm for elm in dataDays if userInput.split('-')[1] in elm][0])
        dataDays=dataDays[start:stop+1]
        cont=1
    elif userInput == 'records':
        print('Available records:')
        dataDays=os.listdir(datapath)
        print(dataDays)
    elif userInput == 'end':
        print("breaking script.")
        dataDays=[]
        cont=0

if cont==1:
    print('Generate graph of day(s) connected stories? (y/n)')
    makeGraph=input().lower()
    if makeGraph=='y':
        makeGraph=True
    else:
        makeGraph=False
    print('Generate histogram of days(s) connected story counts (y/n)?')
    makeHist=input().lower()
    if makeHist=='y':
        makeHist=True
    else:
        makeHist=False
    print("Generate graphs of fully connected subgraphs? If so, specify which here in sequence (ex: 0,1):")
    try:
        makeSub=[int(elm) for elm in input().split(',')]
    except:
        makeSub=[]
    if len(makeSub)>0:
        makeSubGraph=True
    else:
        makeSubGraph=False
    print('Save results (y/n)?')
    saveVal=input().lower()
    if saveVal=='y':
        saveVal=True
    else:
        saveVal=False

for day in dataDays:
    df, G = workDay(day, save=saveVal, fullGraphVis=makeGraph, histVis=makeHist, subGraphs=makeSub, subGraphVis=makeSubGraph)
