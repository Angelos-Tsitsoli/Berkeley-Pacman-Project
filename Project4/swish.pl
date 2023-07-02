beautiful(helen).
beautiful(john).
woman(helen).
woman(katerina).
man(john).
man(peter).
man(timos).
rich(john).
rich(peter).
muscular(peter).
muscular(timos).
kind(timos).
likes(X, Y) :- man(X),woman(Y), beautiful(Y).
likes(katerina,X):-man(X),likes(X,katerina).
likes(helen, X) :- man(X), (kind(X), rich(X); muscular(X), beautiful(X)).
happy(X) :- rich(X).
happy(X) :- man(X), likes(X, women), likes(women, X).
happy(X) :- woman(X), likes(X, men), likes(men, X).
