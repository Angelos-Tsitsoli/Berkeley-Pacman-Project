import heapq       # Me aythn thn entolh tha mporoyme na xrhsimopoihsoyme tis synarthseis heappush(geia eisagogi stoixeivn) opos kai heappop (gia thn afairesh ton stoixeion). Tha mporoyme na xrhsimopoihsoyme dhladh tis etoimes synarthseis poy mas dinontai .

class priorityQueue :
    

    def __init__(pq):
      pq.count = 0 # O metrhths aytos tha metraei to megethos toy soroy , dhladh posa stoixeia tha exei synolika esoterika h lista.
      pq.heap = [] # Arxikopoioyme th lista thn opoia tha xrhsimopoioyse gia thn apothikeysi ton stoixeion gia to priority queue . Sygkekrimena tha apotelesei ton soro ths priority queue kai h opoia oysiastika tha apotelesei mia megalh lista h opoia tha dexetai mikres listes me dyo stoixeia , dhladh ta stoixeia ths sygkekrimenhs listas tha einai listes me stoixeia to priority , kai to string toy poy theloyme na baloyme mesa ston soro.

     
                                                 
    def push(pq, item, priority):   # H synarthsh ayth ayto poy tha kanei einai na eisagei ena stoixeio poy tha ths dosoyme emeis sthn oyra proteraiothtas  me proteraiothta ayth poy tha dosoyme eksisoy emeis.
       for i in pq.heap: # Diatrexoyme thn oyra prokeimenoy na broyme an yparxei stoixeio ston soro to opoio exei to idio item kai to idio priority mayto poy theloyme na baloyme mesa , dioti se periptosi poy yparxei tote na mhn baloyme mesa to stoixeio  kathos yparxei hdh sthn oyra kai tha prokypsei diplotypia.
        if item in i:
          if(i[0]==priority):
            print("This item is already in the priority queue with the exact same priority")
            return
       heapq.heappush(pq.heap,[priority,item])  # Se periptosi poy den prokeitai gia diplotypia tote xrhsimopoioyme thn synarthsh heappush poy mas dinetai etoimh kanontas pio pano to import heapq. H synarthsh ayth (heappush) ayto poy tha kanei einai na sygkrinei to stoixeio poy tha eisagei sto soro me ta ypoloipa stoixeia toy soroy oste na brethei kathe stoixeio sthn sosth toy thesh sto soro (h sygkrish poy tha kanei h synarthsh ayth tha einai metaksy arithmoy me arithmo ), epomenos se periptosi poy eisagame mono symboloseires tha yphrxe problhma kathos den tha mporoyse na ginei sygkrish ton stoixeion toy soroy prokeimenoy na brethoyn sthn sosth thesh toys thesh sto soro (tha htan adynato dhladh na ginei sygkrish symboloseiras me symboloseira). Sto shmeio ayto fainetai h xrhsimothta ths parametroy priority gia thn eisagogh ton symboloseiron . Kathe stoixeio toy soroy tha einai mia mikrh lista h opoia tha diathetei os proto stoixeio thn proteraiothta ths kathe symboloseiras poy tha orisoyme emeis gia kathe stoixeio kai os deytero stoixeio ths mikrhs ayths listas tha exoyme thn symboloseira poy theloyme na baloyme , mayton ton tropo h synarthsh heappush tha sygkrinei ton arithmo poy apotelei thn proteraiothta ths symboloseira px  task'x' (x:enas arithmos) poy bazoyme, me tis alles proteraiothtes ton stoixeion toy soroy kai tha topothetei to stoixeio sthn sosth toy thesh ston soro se sxesh me ta alla stoixeia.
       pq.count=pq.count+1
                                                  


    def pop(pq): # H synarthsh ayth ayto poy tha kanei einai na afairei kathe fora poy tha kaleitai, to stoixeio me thn mikroterh proteraiothta apo thn oyra proteraiothtas kai tha to epistrefei. 
         pq.count=pq.count -1 # Efoson tha afaireitai to stoixeio apo thn oyra proteraiothtas to megethos ths oyras meiontetai kata ena  . 
         __, item=heapq.heappop(pq.heap)#   O tropos poy leitoyrgei h synarthsh heappop einai o ekshs : tha kaleitai apo ena antikeimeno typoy Priority Queue kai tha afairei to stoixeio me thn mikroterh proteraiothta poy brisketai sthn oyra kathos kaleitai h etoimh synarthsh poy mas dinetai heappop h opoia tha kanei akribos ayto to pragma dhladh na afairei to mikrotero stoixeio apo ton soro.Telos me thn entolh return tha epistrepsei to stoixeio
         return item
         
         
                                                


    def isEmpty(pq):     # Se periptosi poy klithei h synarthsh ayth tote tha kanei elegxo oson afora to megethos toy soroy , se periptosi poy einai iso me mhden ayto shmainei oti einai kenh h oyra proteraiothtas.
        
       if pq.count==0:
          return True
       
       return False
        
        
                            
    def update(pq,item,priority): # 
     a_flag=0         # H meablhth ayth oysiastika tha apotelei ena flag opos leei kai to onoma ths , dhladh : se tha ksekinaei me thn timh 0 , se periptosi poy parameinei mhden mexri to telos ths synarthshs tote ayto shmainei oti to stoixeio poy theloyme na kanoyme update den yphrxe katholoy mesa sthn oyra (dhladh den yphrxe to idio item me allh proteraiothta , oyte to idio item me akribos thn idia proteraiothta ) kai ara sthn synexeia tha eisaxthei mesa sthn oyra , ostoso se periptosh poy exei timh ish me 1 to flag ayto shmainei oti synebei kapoia apo tis prohgoymenes energeies , opote den tha eisaxthei opos einai sto telos ths synarthshs alla tha eksetastei apo tis endiameses energeies ths synarthshs kai tha kanonistei analogos.Gia paradeigma se periptosi poy theloyme na baloyme ena stoixeio me item idio me kapoio item ths oyras alla me mikroterh proteraiothta tote sthn oyra den theloyme na exoyme kai ta dyo ayta stoixeia alla mono to ena kai sygkekrimena ayto me thn mikroterh proteraiothta.    
     for i in list(pq.heap):  # Sto shmeio ayto diatrexo ena antigrafo ths basikhs listas poy apotelei ton soro mas dioti sthn synexeia oysiastika kano iteration thn lista kai taytoxrona thn allazo (afairo stoixeia pio sygkekrimena ) epomenos an ayto den ginei se antigrafo ths listas kai ginei sthn lista poy xrhsimopoioyme gia soro tote  dhmioyrgithei problhma , efoson 'peirazo' kathos diatrexo thn lista allazei to megethos ths kai bgazei problhmata (px ena problhma poy moy ebgaze prin otan dokimaza na diatrexo thn lista kai oxi antigrafo ths eno thn dietrexa kai ekana mia afairesh den eftane mexri to teleytaio stoixeio) opote ayto eksyphretei epomenes leitoyrgies.
       if item==i[1]: # Se periptosi poy einai idia ta item tote yparxoyn oi parakato periptoseis.
         a_flag=1 # Bazo ton arithmo ena oste na mhn kanei sthn synexeia push sthn grammh 57 xoris na eksetasei thn periptosi kai prokypsoyn oysiastika diplotypies.
         if(i[0]<=priority): # Se periptosi poy yparxei sthn oyra ena stoixeio idio me ayto poy theloyme na kanoymr update alla ayto sthn oyra exei mikroterh h ish priority apo ayto poy theloyme na kanoyme update tote apla na synexisei na diatrexei thn lista .
           continue
          
           
         if(i[0]>priority):                # Se periptosi poy to stoixeio poy theloyme na kanoyme update ena stoixeio me item poy yparxei hdh sthn oyra alla to priority toy stoixeioy poy theloyme na kanoyme update einai mikrotero apo to priority toy stoixeioy poy yparxei hdh sthn lista tote tha ginoyn ta ekshs:    
            pq.heap.remove([i[0],i[1]]) # Afairo to stoixeio sthn oyra me to idio item kai mikrotero priority .
            pq.count=pq.count-1            # Afairo kata ena to count dioti afaireitai gia ligo to megethos.
            pq.push(item,priority)   # Kano push to stoixeio poy thelo na kano update to opoio eixe to mikrotero priority. Thn diadikasia aythn thn kano gia ola ta stoixeia thw listas poy exoyn idio item kai megalytero priority apo ayto to stoixeio poy thelo na kano update. H push ostoso tha ginei mia fora dioti otan paei na ginei kai deyterh tha ginei elegxos apo thn push an yparxei to idio stoixeio sthn lista opoy tha yparxei kai ara den tha ksanaginei , den tha yparksei dhladh diplotypioa.
            heapq.heapify(pq.heap) # Tha xrhsimopoihsoyme thn etoimh synarthsh heapify h opoia tha balei ta stoixeia ths listas se sosth seira oson afora tis proteraiothtes meta tis parapano metaboles poy kaname sthn lista.
    
     
         
     if(a_flag==0):   #Se periptosi poy den ginei tipota apo tis endiameses energeies tote ayto shmainei oti to stoixeio den yparxei katholoy sthn oyra ara apla tha to baloyme mesa.
       pq.push(item, priority)
       return 
    
       
     

     return
     



def PQSort(list):  #Gia thn synarthsh ayth tha xrhsimopoihsoyme thn klash parapano poy ftiaxthke (PriorityQueue) me merikes metaboles.
    finalheap1=[]
    finalheap2=[] # Tha apotelesei thn lista poy tha epistrepsoyme sto telos oysiastika 
    h2=priorityQueue() #H priority queue .
    for i in list:       # Oysiastika tha paroyme kathe arithmo ths listas poy dinoyme kai tha ton baloyme mesa sto priority queue me morfh mikrhs listas opos kai prin opos kaname parapano aplos to priority tha antiprosopeyei ton arithmo poy tha baloyme sthn oyra proteraiothtas kai tha einai oysiastika ayto poy tha mas noiazei kai to item tha einai aplos o arithmos os symboloseira (den mas endiaferei toso to item edo aplos to bazoyme dioti apotelei orisma sthn parapano priority queue ) . To monadiko poy mas endiaferei tha einai to priority dioti ayto tha einai to stoixeio mas kathe fora poy theloyme na to baloyme mesa sto priority queue , kai  to opoio tha sygkrinoyme me ta alla stoixeia oste na mpei sthn sosth thesh sthn oyra proteraiothtas.
        h2.push(i,i)  #Epomenos tha bazoyme kathe arithmo ths listas poy dosame mazi me ton idio arithmo os symboloseira , se morfh mikrhs listas ayta , mesa ston soro.
    
       
    for c in range(h2.count): #Sthn synexeia analoga me to megethos poy tha exei to heap meta to gemisma poy toy kanoyme , tha kanoyme toses epanalipseis oste na baloyme kanpntas pop ta stoixeia ayta sthn lista finalheap1 .Dhladh kathe fora poy tha ginetai pop apo to soro tha bgainei to stoixeio me to mikrotero priority kai tha topotheteitai sto finalheap1 kati poy exeis os apotelesma na perastoyn ta stoixeia apo ton soro sto finalheap1 me seira ayksoysa. Sto finalheap1 ta stoixeia toy tha einai mikres liste me dyo pragmata kathe mikrh lista to priority (poy mas endiaferei kai apotelei ton arithmo ) kai to item .Sthn synexeia tha spasoyme tis liste aytes.               #for c in range(len(h2.heap)): #@!%$%#!%#$@^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^$@&%@$%@$%@$%
          heapq.heappush(finalheap1,h2.pop())

    
    for iterator in finalheap1: # Sto shmeio ayto tha perasoyme ta stoixeia toy fianlheap1 sto finalheap2 , ostoso tha perasoyme sto fianlheap2 mono to kommati priority to opoio einai kai to mono poy mas endiaferei xoris dhladh thn symboloseira poy bazame sto item h opoia eixe os symboloseira to priority kathe stoixeioy. 
      
      finalheap2.append(iterator)

    return finalheap2  # Epomenos prokyptei mia lista me ta stoixeia ths arxikhs listas poy dosame aplos katanemhmena me ayksoysa seira .
    


def main():
    h1=priorityQueue()
    h1.push("task1", 1)
    h1.push("task1", 2)
    h1.push("task0", 0)
    t=h1.pop()
    print(t)
    t=h1.pop()
    print(t)
    h1.push("task3", 3)
    h1.push("task3", 4)
    h1.push("task2", 0)
    t=h1.pop()
    print(t)
    list=[7,9,-3,5,0,19,17]
    print(PQSort(list))
    
    #extra paradeigma parakato apo piazza:
    #h3=priorityQueue()
    #h3.push("task1", 1)
    #h3.push("task2", 1)
    #h3.push("task0", 0)
    #h3.push("task1", 2)
    #print(h3.heap)
    #t=h3.pop()
    #print(h3.heap)
    #h3.update('task1',0)
    #print(h3.heap)
    
    
    
   

 
#Main Method
if __name__ == '__main__':
  #"This code runs when you invoke the script from the command line"
   main()
  

#onoma:Aggelos Tsitsoli
#A.M.:1115202000200







    






