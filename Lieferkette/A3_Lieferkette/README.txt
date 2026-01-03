Zu dieser Aufgabe gibt es bisher 9 Beispieleingaben:

lieferung00.txt
lieferung01.txt
lieferung02.txt
lieferung03.txt
lieferung04.txt
lieferung05.txt
lieferung06.txt
lieferung07.txt
lieferung08.txt

Wir behalten uns vor, bis zum 1.2.2026 weitere Beispiele zu ergänzen. Außerdem wird bis zu diesem Datum ein Ausgabeformat vorgegeben.

Jede Datei beschreibt eine Lieferung und ist so aufgebaut:

1. Zeile: Die Zeitfrist.

2. Zeile: Die Anzahl m der nötigen Arbeitsschritte.

3. Zeile: m Zahlen. Die i-te Zahl gibt die Anzahl der vorhandenen Werke für den Arbeitsschritt i (1≤ i ≤ m) an.

4. Zeile: Die Anzahl w der Werke.

Die folgenden w Zeilen beschreiben jeweils:die Nummer des Werks;

den in diesem Werk möglichen Arbeitsschritt i.

Nächste Zeile: Die Anzahl der Straßen auf der Straßenkarte.

Die folgenden Zeilen beschreiben die Straßenkarte. Jede einzelne Zeile beschreibt eine Straße zwischen zwei Werken beziehungsweise vom Startpunkt S des LKWs zu einem Werk oder von einem Werk zum Zielpunkt T des LKWs. Ein Werk wird beschrieben durch seine Nummer. Damit wird eine Straße so beschrieben:Ausgangspunkt der Straße;

Endpunkt der Straße;

zum Befahren der Straße benötigte Zeit.

Hier das Beispiel lieferung00.txt:

15
2
3 2
5
1 1
2 1
3 1
4 2
5 2
nStreets 9
S 1 1
S 2 4
S 3 3
1 2 2
1 3 1
2 4 7
3 5 5
4 T 3
5 T 1

In diesem Beispiel gibt es eine Zeitfrist von 15 Zeiteinheiten, außerdem gibt es zwei Arbeitsschritte. Der erste Arbeitsschritt kann in drei Werken erledigt werden (1, 2, 3), der zweite Arbeitsschritt in zwei Werken (4 und 5). Insgesamt sind 9 Straßen angegeben. Beispielsweise kann man vom Startpunkt S aus das Werk 2 in 4 Zeiteinheiten und von Werk 3 aus das Werk 5 in 5 Zeiteinheiten erreichen.