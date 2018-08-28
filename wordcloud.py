from pyecharts import WordCloud

f = open("/home/zeno/part-00000")
line = f.readline()
name, value = [], []
while line:
	name.append(line.split("\t")[0])
	value.append(line.split("\t")[1])
	line = f.readline()

wordcloud = WordCloud(width=1300, height=620)
wordcloud.add("", name, value, word_size_range=[20, 100])
wordcloud.render("wordcloud.html")