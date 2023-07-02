# shopSmart.py
# ------------
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
Here's the intended output of this script, once you fill it in:

Welcome to shop1 fruit shop
Welcome to shop2 fruit shop
For orders:  [('apples', 1.0), ('oranges', 3.0)] best shop is shop1
For orders:  [('apples', 3.0)] best shop is shop2
"""
from __future__ import print_function
import shop


def shopSmart(orderList, fruitShops):          # H synarthsh ayth tha brei to oikonomikotero magazi apo ta diathesima oste na pragmatopoihthei h paraggelia.
    """
        orderList: List of (fruit, numPound) tuples
        fruitShops: List of FruitShops
    """                                         # Ayto poy ginetai sthn synarthsh ayth einai to ekshs: diatrexoyme ta magazia ypologizontas kathe fora to kostos ths paraggelias gia to kathena magazi kai sygkrinoyme to kostos toy magazioy poy eksetazetai kathe fora me to mikrotero kostos se magazi poy exei brethei mexri ekeinh th stigmh.Dhladh kratame to mikrotero kostos mexri stigmhs se mia metablhth kai me mia allh metablhth diatrexoyme ta alla magazia kai briskoyme ta kostoi toys eno parallhla ta sygkrinoyme to kathe ena me to mikrotero kostos , opote se periptosi poy brethei kainoyrgio mikrotero kostos tote enhmeronoyme thn metablhth poy diathrei to mikrotero kostos oste na exei to neo mikrotero kostos poy brhkame.
    totalcostshop2=100000.0         #Arxika apla bazo ena poly megalo arithmo oste na isxyei akribos apo kato to if kai h metablhth poy krataei to mikrotero kostos (totalcostshop2) na parei thn proth ths timh apo ta kosth  tvn katasthmatvn , genika to gegonos oti ebala arxika ena poly megalo arithmo einai apla gia na ginei h arxikopoihsh tvn metablhtvn totalcostshop2 kai shop me tis times apo to proto magazi poy tha eksetastei.
    for shop_variable in fruitShops:        # Sto shmeio ayto diatrexontai ta katasthmata analoga me to posa katasthmata yparxoyn sthn lista katasthmatvn.
        totalcostshop1=shop_variable.getPriceOfOrder( orderList)   #Sto shmeio ayto h metablhth ayth dexetai meso ths synarthshs   getPriceOfOrder() ths klashs FruitShop  to kostos ths paraggelias sto sygkekrimeno magazi.   
        if totalcostshop1<totalcostshop2: # Se periptosi poy to kostos toy sygkekrimenoy katasthmatos poy eksetazetai einai to mikrotero apo to mexri tote mikrotero kostos, tote mikrotero kostos theoreitai to kostos toy katasthmatos poy eksetazetai ekeinh thn stigmh.
            totalcostshop2=totalcostshop1 # Ara to totalcostshop2 poy diathrei thn mikroterh timh tha parei kainoyrgia timh thn timh toy totalcostshop1 poy tha einai to neo mikrotero kostos
            shop=shop_variable  # H metablhth shop ayto poy tha kanei oysiastika einai na krathsei poio katasthma  exei to mikrotero kostos ara kathe fora poy exoyme ena kainoyrgio mikrotero kostos h metablhth shop pairnei kai thn timh toy katasthmatos mayto to kostos allios h metablhth shop diathrei thn timh poy diathetei. Gia paradeigma exoyme th lista [shop1,shop2] arxika to totalcostshop1 tha deixnei sto shop1 kai to totalcostshop2 tha parei thn timh kostoys ths paraggelias sto magazi shop1 , opote kai to shop tha exei thn idia timh me to totalcostshop1 .
        

    return shop    


if __name__ == '__main__':
    "This code runs when you invoke the script from the command line"
    orders = [('apples', 1.0), ('oranges', 3.0)]
    dir1 = {'apples': 2.0, 'oranges': 1.0}
    shop1 = shop.FruitShop('shop1', dir1)
    dir2 = {'apples': 1.0, 'oranges': 5.0}
    shop2 = shop.FruitShop('shop2', dir2)
    shops = [shop1, shop2]
    print("For orders ", orders, ", the best shop is", shopSmart(orders, shops).getName())
    orders = [('apples', 3.0)]
    print("For orders: ", orders, ", the best shop is", shopSmart(orders, shops).getName())
