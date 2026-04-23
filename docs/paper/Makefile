# Build IEEE TPDS paper
# Prerequisites: texlive-latex-recommended texlive-fonts-recommended texlive-publishers texlive-bibtex-extra

MAIN = main
BIB  = references

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(BIB).bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,toc,lof,lot,synctex.gz}
