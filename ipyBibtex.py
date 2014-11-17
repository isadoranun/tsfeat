'''
What                : Integration of LaTeX into IPython (extension)
Author              : Marcin 'zmk' Zamojski
License             : GPLv3
Maintained at       : https://gist.github.com/z-m-k/6080008
IPython Notebook at : http://nbviewer.ipython.org/6080008/ipyBibtex.ipynb


With this extension you can use LATEX style references within IPython (Notebook) while still using markdown for formatting.

Required:

* ipython-dev
* biblio-py
* markdown

++++++++++++++
Usage
++++++++++++++

First, you need to load the extension. Use %load_ext (or better %reload_ext):
In [1]:
  %reload_ext ipyBibtex       

You can specify the max number of co-authors in the output before substituting with 'et. al.,' using %etal. The default value upon loading the extension is 3, i.e. articles with 4+ authors will have 'et. al.'
This needs to be done before %bibFile
In [2]:
	%etal 3 

Next you need to specify a bibliography file you want to use. There are two line magics available for this.

* %bibFile will open the file and prepare the required dictionaries
* %bibFileRobust will do the same and print all keys that were found

In [3]:
	%bibFile path/to/file.bib

Finally you can write your text with in a code cell with %%cite cell magic.

In [4]:
	%%cite
	Lorem ipsum dolor sit amet __\citep{hansen1982,gallanttauchen1996,crealkoopmanlucas2013}__, consectetuer adipiscing elit, sed diam nonummy nibh euismod tincidunt ut laoreet dolore magna aliquam erat volutpat.
	Ut wisi enim ad minim veniam, quis nostrud exerci tation ullamcorper suscipit lobortis nisl ut aliquip ex ea commodo consequat.
	Duis autem vel eum iriure dolor in hendrerit in vulputate velit esse molestie consequat, vel illum dolore eu feugiat nulla facilisis at vero eros et accumsan et iusto odio dignissim qui blandit praesent luptatum zzril delenit augue duis dolore te feugait nulla facilisi.
	Nam liber tempor cum soluta nobis eleifend option congue nihil imperdiet doming id quod mazim placerat facer possim assum.
	 
	Typi non habent claritatem insitam; est usus legentis in iis qui facit eorum claritatem.
	Investigationes demonstraverunt lectores legere me lius quod ii legunt saepius, __\cite{hayashi2000,engle1982}__.

++++++++++++++
Changelog
++++++++++++++
2013-07-15 # v0.1.0 # Load bib files and provide basic substitution for \cite, \citet, and \citep

'''
from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
from IPython.core.displaypub import publish_display_data
from markdown import markdown
from yapbib import biblist
import re

@magics_class
class printTex(Magics):
	''''''

	def __init__(self, shell):
		super(printTex, self).__init__(shell)
		# self.bibliography=None
		self.etal=3
		self.cite={
			't':{},
			'p':{}
		}
		wrapText=re.compile('(?<!\n)\n(?!\n)|[ ]+')
		self.wrapText=lambda x: wrapText.sub(' ', x)
		self.citeRe=re.compile('(?P<all>\\cite(?P<mode>t|p)*\{(?P<cite>[\w\d\s,]*)\})')
		
	# def cite
	@line_magic
	def bibFile(self, line, robust=False):
		'''Opening bibliography file'''
		bibliography=biblist.BibList()
		bibliography.import_bibtex(line, normalize=False)
		for key in bibliography.bib.keys():
			curArt=bibliography.bib[key]
			curAuth=zip(*curArt['author'])[1:3]
			if len(curAuth[0])>1:
				if len(curAuth[0])>self.etal:
					curAuth=curAuth[0][0]+' et. al.'
				else:
					curAuth= ', '.join(curAuth[0][:-1])+' and '+curAuth[0][-1]
			else:
				curAuth=curAuth[0][0]
			self.cite['t'][bibliography.bib[key]['_code']]='{0} ({1})'.format(curAuth, curArt['year'])
			self.cite['p'][bibliography.bib[key]['_code']]='{0}, {1}'.format(curAuth, curArt['year'])
		if robust:
			publish_display_data('printTex',
				{'text/plain': '{0} loaded, there are {1} keys ({2})'.format(line, len(self.cite['t'].keys()), ', '.join(self.cite['t'].keys())),
				'text/html': markdown('_{0}_ loaded<br/>There are __{1}__ keys:<br/>{2}'.format(line, len(self.cite['t'].keys()), ', '.join(self.cite['t'].keys())))})

	@line_magic
	def bibFileRobust(self, line):
		self.bibFile(line, robust=True)

	@line_magic
	def etal(self, line):
		self.etal=int(line)

	@cell_magic
	def cite(self, line, cell):
		if len(line)>0:
			cell=line+'\n'+cell
		for c in self.citeRe.finditer(cell):
			refs=[self.cite['p'][ref.strip()] if c.group('mode')=='p' else self.cite['t'][ref.strip()] 
				for ref in c.group('cite').split(',')]
			cell=cell.replace('\\'+c.group('all'), '; '.join(refs) if c.group('mode')!='p' else '('+'; '.join(refs)+')')
		publish_display_data('printTex',
			{
				'text/html': markdown(cell),
				'text/plain': self.wrapText(cell)
			})
				
def load_ipython_extension(ip):
	#register magic to hold state
	printTexObj = printTex(ip)
	ip.register_magics(printTexObj)