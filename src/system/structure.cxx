#include <string.h>

#include "system/system.h"
#include "system/parameters.h"
#include "system/structure.h"
#include "main/defines.h"
#include "io/io.h"



// Class constructors
Structure::Structure() {
  atomCount=0;

  bondCount=0;
  angleCount=0;
  diheCount=0;
  imprCount=0;
  cmapCount=0;

  shakeHbond=false;

  setup_parse_structure();
}

Structure::~Structure() {
}



// Parsing functions
void Structure::setup_parse_structure()
{
  parseStructure[""]=&Structure::error;
  helpStructure[""]="?> How did we get here?\n";
  parseStructure["reset"]=&Structure::reset;
  helpStructure["reset"]="?structure reset> This deletes the structure data structure.\n";
  parseStructure["file"]=&Structure::file;
  helpStructure["file"]="?structure file psf [filename]> This loads the system structure from the CHARMM PSF (protein structure file)\n";
  parseStructure["shake"]=&Structure::parse_shake;
  helpStructure["shake"]="?structure shake [hbond/none]> Turn hydrogen bond length constraints on or off.\n";
  parseStructure["print"]=&Structure::dump;
  helpStructure["print"]="?structure print> This prints selected contents of the structure data structure to standard out\n";
  parseStructure["help"]=&Structure::help;
  helpStructure["help"]="?structure help [directive]> Prints help on state directive, including a list of subdirectives. If a subdirective is listed, this prints help on that specific subdirective.\n";
}

void parse_structure(char *line,System *system)
{
  char token[MAXLENGTHSTRING];
  std::string name;

  if (!system->structure) {
    system->structure=new Structure();
  }

  io_nexta(line,token);
  name=token;
  if (system->structure->parseStructure.count(name)==0) name="";
  // So much for function pointers being elegant.
  // call the function pointed to by: system->structure->parseStructure[name]
  // within the object: system->structure
  // with arguments: (line,token,system)
  (system->structure->*(system->structure->parseStructure[name]))(line,token,system);
}

void Structure::help(char *line,char *token,System *system)
{
  char name[MAXLENGTHSTRING];
  io_nexta(line,name);
  if (name=="") {
    fprintf(stdout,"?structure > Available directives are:\n");
    for (std::map<std::string,std::string>::iterator ii=helpStructure.begin(); ii!=helpStructure.end(); ii++) {
      fprintf(stdout," %s",ii->first.c_str());
    }
    fprintf(stdout,"\n");
  } else if (helpStructure.count(name)==1) {
    fprintf(stdout,helpStructure[name].c_str());
  } else {
    error(line,name,system);
  }
}

void Structure::error(char *line,char *token,System *system)
{
  fatal(__FILE__,__LINE__,"Unrecognized token after structure: %s\n",token);
}

void Structure::reset(char *line,char *token,System *system)
{
  delete system->structure;
  system->structure=NULL;
}

void Structure::file(char *line,char *token,System *system)
{
  FILE *fp;
  io_nexta(line,token);
  if (strcmp(token,"psf")==0) {
    io_nexta(line,token);
    fp=fpopen(token,"r");
    add_structure_psf_file(fp);
    fclose(fp);
  } else {
    fatal(__FILE__,__LINE__,"Unsupported structure file format: %s\n",token);
  }
}

void Structure::parse_shake(char *line,char *token,System *system)
{
  io_nexta(line,token);
  if (strcmp(token,"hbond")==0) {
    shakeHbond=true;
  } else if (strcmp(token,"none")==0) {
    shakeHbond=false;
  } else {
    fatal(__FILE__,__LINE__,"Unrecognized token %s for structure shake selection. Try hbond or none\n",token);
  }
}

void Structure::dump(char *line,char *token,System *system)
{
  fprintf(stdout,"%s:%d IMPLEMENT Structure::dump function.\n",__FILE__,__LINE__);
}

void Structure::add_structure_psf_file(FILE *fp)
{
  char line[MAXLENGTHSTRING];
  char token[MAXLENGTHSTRING];
  int i,j,k;

  // "read" header"
  fgets(line, MAXLENGTHSTRING, fp);
  io_nexta(line,token);
  if (strcmp(token,"PSF")!=0) {
    fatal(__FILE__,__LINE__,"First line of PSF must start with PSF\n");
  }

  // "Read" title
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of title lines");
  for (i=0; i<j; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
  }

  // Read atoms
  fgets(line, MAXLENGTHSTRING, fp);
  atomCount=io_nexti(line,fp,"psf number of atoms");
  atomList.clear();
  atomList.reserve(atomCount);
  for (i=0; i<atomCount; i++) {
    fgets(line, MAXLENGTHSTRING, fp);
    struct AtomStructure at;
    at.atomIdx=io_nexti(line)-1;
    if (at.atomIdx!=i) {
      fatal(__FILE__,__LINE__,"Found atom %d when atom %d expected\n",at.atomIdx,i);
    }
    at.segName=io_nexts(line);
    at.resIdx=io_nexti(line);
    at.resName=io_nexts(line);
    at.atomName=io_nexts(line);
    at.atomTypeName=io_nexts(line);
    at.charge=io_nextf(line);
    at.mass=io_nextf(line);
    atomList.emplace_back(at);
  }

  // Read bonds
  fgets(line, MAXLENGTHSTRING, fp);
  bondCount=io_nexti(line,fp,"psf number of bonds");
  bondList.clear();
  bondList.reserve(bondCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<bondCount; i++) {
    struct Int2 bond;
    for (j=0; j<2; j++) {
      k=io_nexti(line,fp,"psf bond atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in bond %d is out of range\n",k,i);
      }
      bond.i[j]=k;
    }
    bondList.emplace_back(bond);
  }
  
  // Read angles
  fgets(line, MAXLENGTHSTRING, fp);
  angleCount=io_nexti(line,fp,"psf number of angles");
  angleList.clear();
  angleList.reserve(angleCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<angleCount; i++) {
    struct Int3 angle;
    for (j=0; j<3; j++) {
      k=io_nexti(line,fp,"psf angle atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in angle %d is out of range\n",k,i);
      }
      angle.i[j]=k;
    }
    angleList.emplace_back(angle);
  }
  
  // Read dihes
  fgets(line, MAXLENGTHSTRING, fp);
  diheCount=io_nexti(line,fp,"psf number of dihedrals");
  diheList.clear();
  diheList.reserve(diheCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<diheCount; i++) {
    struct Int4 dihe;
    for (j=0; j<4; j++) {
      k=io_nexti(line,fp,"psf dihedral atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in dihedral %d is out of range\n",k,i);
      }
      dihe.i[j]=k;
    }
    diheList.emplace_back(dihe);
  }
  
  // Read imprs
  fgets(line, MAXLENGTHSTRING, fp);
  imprCount=io_nexti(line,fp,"psf number of impropers");
  imprList.clear();
  imprList.reserve(imprCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<imprCount; i++) {
    struct Int4 impr;
    for (j=0; j<4; j++) {
      k=io_nexti(line,fp,"psf improper dih atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in improper %d is out of range\n",k,i);
      }
      impr.i[j]=k;
    }
    imprList.emplace_back(impr);
  }
  
  // Ignore donors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of donors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf donor atom");
  }
  
  // Ignore acceptors
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf number of acceptors");
  for (i=0; i<2*j; i++) {
    io_nexti(line,fp,"psf acceptor atom");
  }
  
  // Not even sure what this section is...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf nnb???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf nnb???");
  }

  // Or this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf ngrp???");
  for (i=0; i<3*j; i++) {
    io_nexti(line,fp,"psf ngrp???");
  }

  // OR this one...
  fgets(line, MAXLENGTHSTRING, fp);
  j=io_nexti(line,fp,"psf molnt???");
  for (i=0; i<atomCount; i++) {
    io_nexti(line,fp,"psf molnt???");
  }

  // ignore lone pairs
  fgets(line, MAXLENGTHSTRING, fp);
  i=io_nexti(line,fp,"psf lone pairs");
  j=io_nexti(line,fp,"psf lone pair hydrogrens");
  if (i!=0 || j!=0) {
    fatal(__FILE__,__LINE__,"Program is not set up to treat lone pairs. Found NUMLP=%d NUMLPH=%d in psf\n",i,j);
  }
  
  // Read cmaps
  fgets(line, MAXLENGTHSTRING, fp);
  cmapCount=io_nexti(line,fp,"psf number of cmaps");
  cmapList.clear();
  cmapList.reserve(cmapCount);
  fgets(line, MAXLENGTHSTRING, fp);
  for (i=0; i<cmapCount; i++) {
    struct Int8 cmap;
    for (j=0; j<8; j++) {
      k=io_nexti(line,fp,"psf cmap atom")-1;
      if (k>=atomCount || k<0) {
        fatal(__FILE__,__LINE__,"Atom %d in cmap %d is out of range\n",k,i);
      }
      cmap.i[j]=k;
    }
    cmapList.emplace_back(cmap);
  }
}
