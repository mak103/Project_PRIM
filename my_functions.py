import scipy.sparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm

class database:
    
    def __init__(self, data_profile, data_skill):
        self.data_profile = data_profile
        self.data_skill = data_skill
        
    def skill_list(self):
        """Put all the skill names in the skill database in a list
    
            Returns
            -------
            data_skill_list:`list`.shape (len(data_skill),)
                Contains the names of each skill which appears 
                in the skill database
        """
        data_skill_list = []
        for skill in self.data_skill:
            if 'name' in skill.keys():
                data_skill_list.append(skill['name'])
        return data_skill_list
    
    def top_skill_list(self):
        """Search all the top skills in each profile, if it also 
            appears in the skill database, then put this skill
            name in the list
    
            Returns
            -------
            top_skill_list:`list`.
                Contains the names of each skill which appears 
                both in the skill and in the 'Top Skills' part of 
                the profile database      
        """
        data_skill_list = self.skill_list()
        top_skill_list = []
        for i in range(len(self.data_profile)):
            if 'skills' in self.data_profile[i].keys():
                if self.data_profile[i]['skills'][0]['title'] == 'Top Skills':
                    for skills in self.data_profile[i]['skills'][0]['skills']:
                        if skills['title'] in data_skill_list:
                            top_skill_list.append(skills['title'])
        return top_skill_list
    
    def all_skill_list(self):
        """Search all skills in each profile, if it also appears 
            in the skill database, then put this skill name in 
            the list
    
            Returns
            -------
            top_skill_list:`list`.
                Contains the names of each skill which appears 
                both in the skill and in profile database. It 
                includes repetitions.
        """
        data_skill_list = self.skill_list()
        all_skill_list = []
        for i in range(len(self.data_profile)):
            if 'skills' in self.data_profile[i].keys():
                for j in range(len(self.data_profile[i]['skills'])):
                    for skills in self.data_profile[i]['skills'][j]['skills']:
                        if skills['title'] in data_skill_list:
                            all_skill_list.append(skills['title'])
        return all_skill_list
    
    def count_words(self,top_only=True):
        """Count the frequent of the appearence for each skill in 
            the skill_list. Then, set a penality coefficient for
            each skill, which depends on it's frequent
    
            Input
            ----------
            top_only: `boolean`.
                Decicde the input is top_skill_list or all_skill 
                _list
            
            Returns
            -------
            feature:`list`.
                Contains the names of unique skill names in the
                input list.
                
            coff:`float64`,shape (len(feature),).
                Penality coefficient for each skill, the more 
                frequent a skill appears, the smaller cofficient 
                it has. Here I use the log function
        """
        if top_only:
            skill_list = self.top_skill_list()
        else:
            skill_list = self.all_skill_list()
        word_counts = Counter(skill_list)
        top_n = word_counts.most_common(len(word_counts))
        feature = []
        proportion = []
        for i in top_n:
            feature.append(i[0])
            proportion.append(i[1])
        coff = 1./(np.log(proportion)+1)
        return feature, coff

    def build_top_skill_frame(self, binary=False ):
        """Bulid a DataFrame by the top skills in each profile.
    
            Parameters
            ----------
            binary: `boolean`.
                Decicde the DataFrame contains whether the appearence 
                of each skill(0 means this profile doesn't have this
                skill, 1 means otherwise) or the endoresementCount of
                each skill.
              
            Inputs
            ----------
            feature:`list`.
                Contains the names of unique skill names.
                
            coff:`float64`,shape (len(feature),).
                Penality coefficient for each skill, the more frequent 
                a skill appears, the smaller cofficient it has.
            
            Returns
            -------
            df:`DataFrame`,shape(len(data_profile), len(feature)).
                Each row corresponds to a profile. The top skill
                information are stored by the binary number or
                the endoresementCount values.
                
        """
        feature, coff = self.count_words(top_only=True)
        array = scipy.sparse.lil_matrix((len(self.data_profile), len(feature)))
        for i in tqdm(range(len(self.data_profile))):
            rang = np.zeros(len(feature))
            if 'skills' in self.data_profile[i].keys():
                for skills in self.data_profile[i]['skills']:
                    if self.data_profile[i]['skills'][0]['title'] == 'Top Skills':
                        for skill in self.data_profile[i]['skills'][0]['skills']:
                            if skill['title'] in feature:
                                if 'endoresementCount' in skill.keys():
                                    if '+' in skill['endoresementCount']:
                                        count = 100
                                    else:
                                        count = int(skill['endoresementCount'])
                                    index = feature.index(skill['title'])
                                    array[i,index] = count * coff[index]
        df = pd.DataFrame(data=array.A, columns=feature)
        if binary:
            df = (df != 0).astype('int')
        return df
    
    def build_all_skill_frame(self, binary=False, top_effect=10):
        """Bulid a DataFrame by all the skills in each profile.
    
           Parameters
           ----------
           binary: `boolean`.
                Decicde the DataFrame contains whether the appearence 
                of each skill(0 means this profile doesn't have this
                skill, 1 means otherwise) or the endoresementCount of
                each skill.
                
           top_effect: `int`.
               It refers to the importance of the top skills, comparing
               with other skills
            
           Inputs
           ---------- 
           feature:`list`.
                Contains the names of unique skill names.
                
           coff:`float64`,shape (len(feature),).
                Penality coefficient for each skill, the more frequent 
                a skill appears, the smaller cofficient it has.
            
            Returns
            -------
            df:`DataFrame`,shape(len(data_profile), len(feature)).
                Each row corresponds to a profile. The skill information 
                are stored by the binary number or the endoresementCount 
                values.
                
        """
        feature, coff = self.count_words(top_only=False)
        array = scipy.sparse.lil_matrix((len(self.data_profile), len(feature)))
        effect = 1
        for i in tqdm(range(len(self.data_profile))):
            rang = np.zeros(len(feature))
            if 'skills' in self.data_profile[i].keys():
                for skills in self.data_profile[i]['skills']:
                    for j in range(len(self.data_profile[i]['skills'])):
                        if self.data_profile[i]['skills'][j]['title'] == 'Top Skills':
                            effect = top_effect
                        else:
                            effect = 1
                        for skill in self.data_profile[i]['skills'][j]['skills']:
                            if skill['title'] in feature:
                                if 'endoresementCount' in skill.keys():
                                    if '+' in skill['endoresementCount']:
                                        count = 100
                                    else:
                                        count = int(skill['endoresementCount'])
                                    index = feature.index(skill['title'])
                                    array[i,index] = count * coff[index] * effect
        df = pd.DataFrame(data=array.A, columns=feature)
        if binary:
            df = (df != 0).astype('int')
        return df

    def show_skill(self, index, show_other_skill=False):
        """Print a profile's skills.
    
           Parameters
           ----------
           index: `int`.
               The index of the profile.
                
           show_other_skill: `boolean`.
               Decide whether this function print only top skills or 
               print all the skills              
        """
        show = [[],[]]
        show[0].append('Top Skills: ')
        show[1].append('Other Skills: ')
        if 'skills' not in self.data_profile[index].keys():
            print('This profile doesn\'t contain any skill.')
            return
        for skills in self.data_profile[index]['skills']:
            if skills['title'] == 'Top Skills':
                for skill in skills['skills']:
                    show[0].append(skill['title'])
            else:
                for skill in skills['skills']:
                    show[1].append(skill['title'])
        print(show[0])
        if show_other_skill:
            print(show[1])
        return