import csv
import os

def cls():
    os.system('cls' if os.name=='nt' else 'clear')


'''rawTicketList is a list of dicionary, where every ticket is its own dict
with all the original data'''
rawTicketList=[]

'''ticketList is a list of dictionaries, but only including the data we need
(ticket num, location, subject, and request detail)'''
ticketList=[]

with open('./Raw Data/TicketsPt1.csv','r') as dataFile:
    reader = csv.DictReader(dataFile)
    for row in reader:
        rawTicketList.append(row)
        tempDict={}
        tempDict['No.']=row['No.']
        tempDict['Location']=row['Location']
        tempDict['Subject']=row['Subject']
        tempDict['Detail']=row['Request Detail']
        ticketList.append(tempDict)

''' this loops removes ticket that included a forwarded message or a reply'''

tempList =[]
altCount =0
for ticket in ticketList:
    if (ticket['Subject'][0:3]=='Re:' or ticket['Subject'][0:4]=='Fwd:'):
        altCount+=1
    else:
        tempList.append(ticket)

ticketList = tempList

''' here we label the tickets as level 1 or level 2. If level 1, target is 0
if level 2, target is 1
if not sure, labeled .5

not sure:
email access (level 2)
non discriptive Tickets (level 1)
IT student phone line (level 1)
post purchase tickets (level 1)
network hardware issues (level 2)
tickets that eventually lead to a purchase, but not described in original ticket (level 1)


Tickets labeled 0:
PW
Email trees
HP hardware/software issues
Sound HP issue
Email access/email diactivation due to no longer working
School phone/phone extension issues
cameras*
app installations
creating accounts
google voice
projectors
imaging laptops/ipads
Cables/other necessary hardware
onboarding/new staff
events setup
tech depots
ipads
laptop replacements/loaners
student printing
Chromebook hardware/software issues
adding printers/fixing printers/getting cotg to come
zoom licesing
ID printing support
bells
MM scanners
dayforce Login
PS login for STUDENTS
product activation for MS apps
Google voice
BIG IP/SSM/Impact (change to level 1)

Tickets labeled 1:
changes to large scale account restrictions (youtube, websites, OU settings, extensions)
vpn access

large scale wifi issues
large scale phone issues
clock purchases/fix
cameras issues/installations
FE backup
phone purchases (explicit)
reaching out to vendors (verizon)
PS/Tableau access
adding devices (bought by schools) to network


*** takes about 1 hr to label about 500 tickets***

'''

for i in range(2251, len(ticketList)-1):
    if (i <3001):
        cls()
        print('TICKET #:')
        print(str(ticketList[i]['No.']))
        print('Currently on Ticket: ' + str(i))
        print('SUBJECT: ')
        print(str(ticketList[i]['Subject']))
        print('--------------------')
        print('DETAIL: ')
        print(str(ticketList[i]['Detail']))
        print('--------------------')
        targetInput = input("enter target: ")
        if (targetInput == 'end'):
            break
        ticketList[i]['Target'] = targetInput
    else:
        ticketList[i]['Target'] = '-1'



'''creating a new csv file with only the 5 categories mentioned before that are important plus the target
(ticket num, location, subject, and ticket detail)'''
fieldnames =['No.', 'Location','Subject', 'Detail','Target']
with open('ticketTrainBryan.csv','w',newline='') as newFile:
    writer = csv.DictWriter(newFile, fieldnames=fieldnames)
    writer.writeheader()
    for ticket in ticketList:
        writer.writerow(ticket)
