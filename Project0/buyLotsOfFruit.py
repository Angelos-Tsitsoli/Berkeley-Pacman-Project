# buyLotsOfFruit.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
To run this script, type

  python buyLotsOfFruit.py

Once you have correctly implemented the buyLotsOfFruit function,
the script should produce the output:

Cost of [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)] is 12.25
"""
from __future__ import print_function

fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75,
               'limes': 0.75, 'strawberries': 1.00}


def buyLotsOfFruit(orderList):
    """
        orderList: List of (fruit, numPounds) tuples

    Returns cost of order
    """
    totalCost = 0.0
    for order_iterator in orderList:    # H metablhth order iterator tha diatreksei to orderList prokeimenoy na ypologisei to synoliko kostos ths paraggelias . Oi times poy dinontai os times gia ta froyta as ypothesoyme oti kathe tetoia timh antistoixei sto ena kilo toy eidoys toy froytoy gia paradeigma dinetai oti ta mhla exoyn timh 2.00 opote ayto tha to diabasoyme os ekshs: ta mhla exoyn timh 2.00 $ to kilo.
        if(order_iterator[0]=='apples'):       # Gnorizoyme oti ta mhla exoyn timh 2.00 $ (to kilo) , ara efoson kai mono an yparxoyn mila sthn paraggelia  tote pollaplasiazoyme ton arithmo tvn kilvn apo mhla poy periexei h paraggelia me thn timh toy enos kiloy toy sygkekrimenoy froytoy , gia paradeigma h paraggelia  anagrafei 2 kila mhla ara to kostos toys tha einai to ekshs : efoson to ena kilo kostizei 2.00$ tote ta dyo kila tha kostizoyn 4.00$.
            totalCost=totalCost+order_iterator[1]*2.00  #Omoivs me ta mhla tha kai parakatv me ta ypoloipa froyta , tha ginei prvta elegxos an yparxei to sygkekrimeno froyto sthn paraggelia kai sthn synexeia analoga ta kila toy sygkekrimenoy froytoy poy zhtoyntai tha ypologistei to kostos toys.
        if(order_iterator[0]=='pears'):                 # Kathe fora poy ypologizetai to kostos kapoioy froytoy athroizetai me to yparxon synoliko kostos poy exei ypologistei mexri ekeinh thn stigmh , prokeimenoy sto telos na exoyme to synoliko kostos .
            totalCost=totalCost+order_iterator[1]*1.75
        if(order_iterator[0]=='oranges'):
            totalCost=totalCost+order_iterator[1]*1.50
        if(order_iterator[0]=='limes'):
            totalCost=totalCost+order_iterator[1]*0.75   
        if(order_iterator[0]=='strawberries'):
            totalCost=totalCost+order_iterator[1]*1.00
    return totalCost  # Telos epistrefoyme to synoliko kostos.


# Main Method
if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orderList = [('apples', 2.0), ('pears', 3.0), ('limes', 4.0)]
    print('Cost of', orderList, 'is', buyLotsOfFruit(orderList))
