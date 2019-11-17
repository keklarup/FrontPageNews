#Goal: Capturing images from the newspaper daily headlines displayed by the Newseum. May be a gray area, legally, as Newseum says I need to contact the newspaper publisher directly. But we'll see...

#http://www.newseum.org/todaysfrontpages/?tfp_show=all

    
from urllib.request import urlopen
from time import gmtime, strftime
from bs4 import BeautifulSoup
import os
import time
import urllib.request
import random

print('Going to Newseum website and counting how many front pages are currently available...')
t=strftime("%Y%m%d") #today
url='http://www.newseum.org/todaysfrontpages/?tfp_show=all' #today
html=urlopen(url) #download page
soup=BeautifulSoup(html.read(),'html.parser') #prep Newseum page for parsing
soup2=soup.findAll('script') #finding location of script section of page
l=-1;
loc=-1
#longest script section is the one with all the newspaper front pages. The following code isolates that section.
for x in range(0,len(soup2)):
    if len(str(soup2[x]))>l:
        l=len(str(soup2[x]))
        loc=x
# print(loc,l) #prints out location of section with longest length, where the front pages reside.
#Going through the front page section and finding all the links to PDFs of front pages:
soup_string=str(soup2[loc])
soup_str_split=soup_string.split(",")
#getting the links for the pdfs:
soup_pdf_links=[elm[6:].replace('"','').replace("\\/","/") for elm in soup_str_split if elm.startswith('"pdf')]
#Display results:
print('Today is:')
print(t)
print('Current timestamp on Newseum data (early AM can still have prev. day):')
print(set([elm.split('/')[-2] for elm in soup_pdf_links]))
print('Number of front pages currently found on Newseum website:')
print(len(soup_pdf_links))
print('continue (y/n)?')
check=str(input()).lower()
if check=='y':
    #Check if directory for saving results already exists. If not, make new one:
    try:
        os.listdir(os.path.join("data",t))
        print('directory already exists. Checking if some headlines already found...')
        print(len(os.listdir(os.path.join("data",t))), 'headlines found. Updating list of headlines to find...')
        soup_pdf_links=[link for link in soup_pdf_links if link.split('/')[-1] not in ['_'.join(elm.split('_')[1:]) for elm in os.listdir(os.path.join("data",t))]]
        print('New size of set to find:',len(soup_pdf_links))
    except:
        print('making new directory')
        os.mkdir(os.path.join("data",t))
    #looping through all the front page links
    for x in range(0,len(soup_pdf_links)):
        try:
            url=soup_pdf_links[x]
            if strftime("%d")[0]=='0':
                n=url.split('/pdf%s/'%(strftime("%d").replace('0','')))[1]#today
            else:
                n=url.split('/pdf%s/'%(strftime("%d")))[1]#today
            label=t+"_"+n
            t1=time.time()
            urllib.request.urlretrieve(url, 'data/%s/%s'%(t,label))
            t2=time.time()
            #print(url,' took', t2-t1, 'seconds.')
            wait=random.randrange(10,30,1)*.1;
            time.sleep(wait) #to not abuse the newseum's servers, adding a delay. Making it a bit random to play with mimicing a human.
            if x%10==0:
                print(x,n)
        except:
            print('something went wrong!') 
else:
    print('Stopping early')