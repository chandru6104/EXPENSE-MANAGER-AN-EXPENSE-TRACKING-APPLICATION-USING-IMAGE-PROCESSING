
from PIL import Image
import pytesseract as lss
import argparse
import cv2
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np  
import csv
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet 
limit=10000
grand_total = 0
# nltk.download('punkt',quiet=True)
# nltk.download('wordnet',quiet=True)
lss.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
# path = 'bill.jpeg'
# path = 'restarunt_bill1.jpg'
# path = 'bill2.webp'

# with open('entertainment.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('investment.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('shopping.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('grocery.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('transport.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('home.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])
# with open('miscellaneous.csv', 'a', newline='') as csvfile:
#     spamwriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
#     spamwriter.writerow(['date','organisation','amount'])


grocery_list = str("MILK,BREAD,BISCUIT"
        "BREAD,MILK,BISCUIT,CORNFLAKES"
        "BREAD,TEA,BOURNVITA"
        "JAM,MAGGI,BREAD,MILK"
        "MAGGI,TEA,BISCUIT"
        "BREAD,TEA,BOURNVITA"
        "MAGGI,TEA,CORNFLAKES"
        "MAGGI,BREAD,TEA,BISCUIT"
        "JAM,MAGGI,BREAD,TEA"
        "BREAD,MILK"
        "COFFEE,COCK,BISCUIT,CORNFLAKES"
        "COFFEE,COCK,BISCUIT,CORNFLAKES"
        "COFFEE,SUGER,BOURNVITA"
        "BREAD,COFFEE,COCK"
        "BREAD,SUGER,BISCUIT"
        "COFFEE,SUGER,CORNFLAKES"
        "BREAD,SUGER,BOURNVITA"
        "BREAD,COFFEE,SUGER"
        "BREAD,COFFEE,SUGER"
        "TEA,MILK,COFFEE,CORNFLAKES"
        )
grocery_list = grocery_list.split(',')
grocery_list = [l.lower() for l in grocery_list]
# print(grocery_list)

def detection_sys(path):
    #nested list of image rgb codes
    img = cv2.imread(path)
    #pytesserect predict the rgb coorordinates and give the text
    text = lss.image_to_string(img) 
    print(text)
    #set bill total as none
    total_amount = None
    x = 0
    # split the text on the newline and return an list of texts
    for line in text.split('\n'):
        if  'Total' in line or 'TOTAL' in line:
            total_amount = line
    if total_amount:  
        try:   
            x = total_amount.split(' ')[-1]
            print(x)
            x = x.replace(',','')
            x = x.replace('$','')
            x = x.replace('£','')
            x = x.replace('₹','')
            x = x.replace('€','')
            x = float(x)
            print(x)
        except:
            try:
                price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
                price = list(map(float,price)) 
                print(max(price))
                x=max(price) 
            except:
                print('Total can\'t found')
    else:
        print('total can\'t detect')
    match=re.findall(r'\d+[/.-]\d+[/.-]\d+', text)
    st=" "
    st=st.join(match)
    print(st)


    # price=re.findall(r'[\$\£\€](\d+(?:\.\d{1,2})?)',text)
    # price = list(map(float,price)) 
    # print(max(price))
    # x=max(price)  

    # print()

    sent_tokens=nltk.sent_tokenize(text)
    print(sent_tokens)
    head = sent_tokens[0].splitlines()[0]
    print(head)
    tokenizer = nltk.RegexpTokenizer(r"\w+")           
    new_words = tokenizer.tokenize(text)
    print('new_words',new_words)
    stop_words = set(nltk.corpus.stopwords.words('english')) 
    print('stop_words',stop_words)
    filtered_list = [w.lower() for w in new_words if w not in stop_words]
    print('filter_list',filtered_list)

    #entertainment
    entertainment = []
    for syn in wordnet.synsets("entertainment"): 
        for l in syn.lemmas(): 
            entertainment.append(l.name()) 
    l=['happy','restaurant','food','kitchen','hotel','room','park','movie','cinema','popcorn','combo meal']
    entertainment=entertainment+l
    home_utility=[] 
    for syn in wordnet.synsets("home"): 
        for l in syn.lemmas(): 
            home_utility.append(l.name()) 
    l2=['internet','telephone','elecricity','meter','wifi','broadband','consumer','reading','gas','water','postpaid','prepaid']
    home_utility+=l2
    
    grocery=[] 
    for syn in wordnet.synsets("grocery"): 
        for l in syn.lemmas(): 
            grocery.append(l.name())
    l3=['bigbasket','milk','atta','sugar','suflower','oil','bread','vegetabe','fruit','salt','paneer','soda']
    grocery+l3         
    grocery + grocery_list

    investment=[] 
    for syn in wordnet.synsets("investment"): 
        for l in syn.lemmas(): 
            investment.append(l.name()) 
    l1=['endowment','grant','loan','applicant','income','expenditure','profit','interest','expense','finance','property','money','fixed','deposit','kissan','vikas']
    investment=investment+l1

    transport=[]
    for syn in wordnet.synsets("car"): 
        for l in syn.lemmas(): 
            transport.append(l.name()) 
    l4=['cab','ola','uber','autorickshaw','railway','air','emirates','aerofloat','taxi','booking','road','highway']
    transport+=l4

    shopping=[]
    for syn in wordnet.synsets("dress"): 
        for l in syn.lemmas(): 
            shopping.append(l.name()) 
    l4=['iphone','laptop','saree','max','pantaloons','westside','vedic','makeup','lipstick','cosmetics','mac','facewash','heels','crocs','footwear','purse']
    shopping+=l4
    e=inv=g=s=t=h=False

    for word in filtered_list:
        if word in entertainment:
            e=True
            break
        elif word in investment:
            inv=True
            break
        elif word in grocery:
            g=True
            break
        elif word in shopping:
            s=True
            break
        elif word in transport:
            t=True
            break
        elif word in home_utility:
            h=True
            break
                


    if(e):
        print("entertainment category")
        category_name = 'entertainment category'

        filename='{}.csv'.format('entertainment')
        #df=pd.read_csv('entertainment.csv')
    elif(inv):
        print("investment category")
        category_name = 'investment category'
        filename='{}.csv'.format('investment')
        #df=pd.read_csv('investment.csv')
    elif(s):
        print("shopping category")
        category_name = 'shopping category'
        filename='{}.csv'.format('shopping')
        #df=pd.read_csv('shopping.csv')
    elif(g):
        print("grocery category")
        category_name = 'grocery category'
        filename='{}.csv'.format('grocery')
        #df=pd.read_csv('grocery.csv')
    elif(t):
        print("transport category")
        category_name = 'transport category'
        filename='{}.csv'.format('transport')
        #df=pd.read_csv('transport.csv')
    elif(h):
        print("home utility category")
        category_name = 'home utility category'
        filename='{}.csv'.format('home')
        #df=pd.read_csv('home.csv')
    else:
        print("miscellaneous")
        category_name = 'miscellaneous'
        filename='{}.csv'.format('miscellaneous')
        #df=pd.read_csv('miscellaneous.csv')

    row_contents = [st,head,x]
    from csv import writer
    
    def append_list_as_row(file, list_of_elem,category_name):
    
        with open(file, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)

    append_list_as_row(filename, row_contents,category_name)


def visualize():
    
    import pandas as pd
    entertainment=pd.read_csv('entertainment.csv')
    investment=pd.read_csv('investment.csv')
    shopping=pd.read_csv('shopping.csv')
    grocery=pd.read_csv('grocery.csv')
    transport=pd.read_csv('transport.csv')
    other=pd.read_csv('miscellaneous.csv')
    home=pd.read_csv('home.csv')


    print(entertainment.head())

    category=['entertainment','investment','shopping','grocery','transport','home','miscellaneous']
    total_entertainment=entertainment['amount'].sum()
    total_investment=investment['amount'].sum()
    total_shopping=shopping['amount'].sum()
    total_grocery=grocery['amount'].sum()
    total_transport=transport['amount'].sum()
    total_home=home['amount'].sum()
    total_miscellaneous=other['amount'].sum()
    amount=[total_entertainment,total_investment,total_shopping,total_grocery,total_transport,total_home,total_miscellaneous]
    print(amount)
    data={'category':category,'total':amount}

    df = pd.DataFrame(data) 

    print(df.head(10))
    # Clean the 'total' column to contain only numeric values
    df['total'] = pd.to_numeric(df['total'], errors='coerce')

# Removing rows with NaN in the 'total' column
    df = df.dropna(subset=['total'])

# Now, the 'total' column should contain only numeric values, and you can proceed with plotting
    plt.pie(df['total'], labels=df['category'], autopct='%1.1f%%', shadow=True, startangle=140)
    plt.show()

    grand_total = df['total'].sum()




    total_row  = {'category':'Total Expenditure','total':grand_total}
    total_row = pd.DataFrame({
        'category':['Total Expenditure'],'total':[grand_total]
    })
    total_df = pd.concat([df,total_row],ignore_index=True)
    total_df.to_csv('total.csv',index=False)


    plt.pie(df['total'], labels=df['category'], autopct='%1.1f%%', shadow=True, startangle=140)
    plt.title(" expenditure")

    df.plot(x='category',y='total',kind='barh',title='Expenditure')
    plt.show()


    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(df['total'], wedgeprops=dict(width=0.5), startangle=-40)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
            bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(category[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                    horizontalalignment=horizontalalignment, **kw)
    plt.show()

    ordered_df = df.sort_values(by='total')
    data_sort = ordered_df

    data_sort.plot(x='category',y='total',kind='barh',title='entertainment expenditure')
    percent=[]
    for i in df['total']:
        per=int((i/df['total'].sum())*100)
        percent.append(per)

    percent.sort()

    ordered_df = df.sort_values(by='total')
    data_sort = ordered_df
    #line chart
    df.plot(x='category', y='total', kind='line', marker='o', title='Expenditure Over Time')
    plt.xlabel('Category')
    plt.ylabel('Total Expenditure')
    plt.show()

    for i in range(len(df)):
        print("{}%  of your expenditure in {} category".format(percent[i],data_sort['category'].iloc[i]))

    print(grand_total)
    if grand_total > limit:
        print('Caution! You have reached the set limit. Avoid spending on unwanted things.')



#start of the loop
folder_path = 'bills'
bill_list = os.listdir(folder_path)
# print(bill_list)

for bill in bill_list:
    bill_path = os.path.join(folder_path,bill)
    bill_path = f'bills/{bill}'
    #bills\online-receipt-maker-and-design-food-templates_wo_Image.png
    if os.path.isfile(bill_path):
        print(f'bill_name:{bill_path}')
        path = bill_path
        detection_sys(path)
visualize()

