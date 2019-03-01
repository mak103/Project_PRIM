import scipy.sparse
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tqdm import tqdm
from mtranslate import translate
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import dill as pickle
from gensim.models import word2vec
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import heapq

class Profile_matching:
    
    def __init__(self, data_profile, data_skill):
        self.data_profile = data_profile
        self.data_skill = data_skill
        
    def data_skill_list(self):
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
    
            -------
            self.top_skill_list:`list`.
                Contains the names of each skill which appears 
                both in the skill and in the 'Top Skills' part of 
                the profile database      
        """
        data_skill_list = self.data_skill_list()
        self.skill_list = []
        for i in range(len(self.data_profile)):
            if 'skills' in self.data_profile[i].keys():
                if self.data_profile[i]['skills'][0]['title'] == 'Top Skills':
                    for skills in self.data_profile[i]['skills'][0]['skills']:
                        if skills['title'] in data_skill_list:
                            self.skill_list.append(skills['title'])
        return
    
    def all_skill_list(self):
        """Search all skills in each profile, if it also appears 
            in the skill database, then put this skill name in 
            the list
    
            -------
            self.all_skill_list:`list`.
                Contains the names of each skill which appears 
                both in the skill and in profile database. It 
                includes repetitions.
        """
        data_skill_list = self.data_skill_list()
        self.skill_list = []
        for i in range(len(self.data_profile)):
            if 'skills' in self.data_profile[i].keys():
                for j in range(len(self.data_profile[i]['skills'])):
                    for skills in self.data_profile[i]['skills'][j]['skills']:
                        if skills['title'] in data_skill_list:
                            self.skill_list.append(skills['title'])
        return 
    
    def count_words(self,top_only=True):
        """Count the frequent of the appearence for each skill in 
            the skill_list. Then, set a penality coefficient for
            each skill, which depends on it's frequent
    
            Input
            ----------
            top_only: `boolean`.
                Decicde the input is top_skill_list or all_skill 
                _list
            
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
            self.top_skill_list()
        else:
            self.all_skill_list()
        word_counts = Counter(self.skill_list)
        top_n = word_counts.most_common(len(word_counts))
        self.feature = []
        proportion = []
        for i in top_n:
            self.feature.append(i[0])
            proportion.append(i[1])
        self.coff = 1./(np.log(proportion)+1)
        return 

    #def build_top_skill_frame(self, binary=True):
    def fit(self, binary=True):
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
            self.feature:`list`.
                Contains the names of unique skill names.
                
            self.coff:`float64`,shape (len(self.feature),).
                Penality coefficient for each skill, the more frequent 
                a skill appears, the smaller cofficient it has.
            
            -------
            self.df:`DataFrame`,shape(len(data_profile), len(self.feature)).
                Each row corresponds to a profile. The top skill
                information are stored by the binary number or
                the endoresementCount values.
                
        """
        self.count_words(top_only=True)
        array = scipy.sparse.lil_matrix((len(self.data_profile), len(self.feature)))
        for i in range(len(self.data_profile)):
            rang = np.zeros(len(self.feature))
            if 'skills' in self.data_profile[i].keys():
                for skills in self.data_profile[i]['skills']:
                    if self.data_profile[i]['skills'][0]['title'] == 'Top Skills':
                        for skill in self.data_profile[i]['skills'][0]['skills']:
                            if skill['title'] in self.feature:
                                if 'endoresementCount' in skill.keys():
                                    if '+' in skill['endoresementCount']:
                                        count = 100
                                    else:
                                        count = int(skill['endoresementCount'])
                                    index = self.feature.index(skill['title'])
                                    array[i,index] = count * self.coff[index]
        self.df = pd.DataFrame(data=array.A, columns=self.feature)
        if binary:
            self.df = (self.df != 0).astype('int')
        return
    
    def build_all_skill_frame(self, binary=True, top_effect=10):
        """Bulid a DataFrame by all the skills in each profile.
          Would take long time, not recommended.
    
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
           self.feature:`list`.
                Contains the names of unique skill names.
                
           self.coff:`float64`,shape (len(self.feature),).
                Penality coefficient for each skill, the more frequent 
                a skill appears, the smaller cofficient it has.
            
            Returns
            -------
            df:`DataFrame`,shape(len(data_profile), len(self.feature)).
                Each row corresponds to a profile. The skill information 
                are stored by the binary number or the endoresementCount 
                values.
                
        """
        self.count_words(top_only=False)
        array = scipy.sparse.lil_matrix((len(self.data_profile), len(self.feature)))
        effect = 1
        for i in tqdm(range(len(self.data_profile))):
            rang = np.zeros(len(self.feature))
            if 'skills' in self.data_profile[i].keys():
                for skills in self.data_profile[i]['skills']:
                    for j in range(len(self.data_profile[i]['skills'])):
                        if self.data_profile[i]['skills'][j]['title'] == 'Top Skills':
                            effect = top_effect
                        else:
                            effect = 1
                        for skill in self.data_profile[i]['skills'][j]['skills']:
                            if skill['title'] in self.feature:
                                if 'endoresementCount' in skill.keys():
                                    if '+' in skill['endoresementCount']:
                                        count = 100
                                    else:
                                        count = int(skill['endoresementCount'])
                                    index = self.feature.index(skill['title'])
                                    array[i,index] = count * self.coff[index] * effect
        self.df = pd.DataFrame(data=array.A, columns=self.feature)
        if binary:
            self.df = (self.df != 0).astype('int')
        return self.df

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
        print("index:",index)
        print(show[0])
        if show_other_skill:
            print(show[1])
        print("  ")
        return
        
    def translate(self):
        """Translate the skills which are written in other languages
            into English. (The result has already stored in the 
            "translation_fr.json")
    
           Output:
           ----------
           translation: `dict`.
               The dictionary. It can translate skill names which are 
               appared in the profile database into English.
               
        """
        translation = {}
        catalog = []
        for i in tqdm(range(len(self.data_profile))):
            if 'skills' in (self.data_profile[i].keys()):
                for skills in self.data_profile[i]['skills']:
                    if self.data_profile[i]['skills'][0]['title'] == 'Top Skills':
                        for skill in self.data_profile[i]['skills'][0]['skills']:
                            title = skill['title'].lower()
                            if title not in catalog:
                                catalog.append(title)
                                result = translate(title, to_language='en').lower()
                                if result != title:
                                    translation[title] = result
        return translation
    
    def match(self, profile_index, nb_profile=3, show_other_skill=True):
        """Matching similar profiles of a given profile (it's index).
    
           Input:
           ----------
           profile_index: `int`.
               The index of a profile, which we want ot match other profiles with.
               
           nb_profile: `int`.
               The number of similar profiles we want to match.
           
           show_other_skill: `bool`.
               Decide if show all the skills in profiles or just their top skills.
        """
        sample = self.df.loc[profile_index].values
        if sum(sample) == 0:
            print('This profile doesn\'t have skill.')
            return 
        score = self.df.values.dot(sample)
        score = list(score)
        self.max_index = heapq.nlargest(nb_profile+1, range(len(score)), score.__getitem__)
        print("input profile:")
        self.show_skill(profile_index, show_other_skill=show_other_skill)
        print("matched profiles:")
        for i in range(len(self.max_index)):
            if i == nb_profile:
                break
            if self.max_index[i] != profile_index:
                self.show_skill(self.max_index[i], show_other_skill=show_other_skill)
        return
    
   
    
class Skill_culstering:
    
    def __init__(self, filename, modelname, dim_model=100):
        """initial the class, form a dataframe from word2vec model
        
           Parameters
           ----------
           filename: `str`.
               Path to a skill list.
               
           modelname: `str`.
               Path to a trained word2vec model.
               In this model, each skill title was a "dim_model" dimension vector
               
           dim_model: `int`.
               Dimension of the vector of the Word2vec model, default 100.
           
           self.data: `DataFrame`.
               A matrix that contains all the vectors of skill titles.
               
           self.skill: `list`.
               A list of skill titles.
               
           self.cluster: `list`.
               Index of items in each cluster.
        """
        self.dim = dim_model
        model = word2vec.Word2Vec.load(modelname) # modelname = '../Utils/word2vec_model_allskills'
        f = open(filename,"rb") # filename = "../Data/all_top_skills_final_fre.txt"
        key_list = pickle.load(f)
        f.close()
        self.data = pd.DataFrame(columns=[np.zeros([self.dim])]) ## how to determine the dimension here
        j = 0
        self.skill = []
        for i in key_list.keys():
            try: 
                self.data.loc[j] = model[i]
                j+=1
                self.skill.append(i)
            except:
                j = j
        self.cluster = []
    
    
    def skill_select(self):
        """choose a skill title in clusters to present this cluster
        
           Parameters
           ----------
           self.present_skill: `list`
               contains skill titles' index which can present it's cluster 
        """
        self.present_skill = []
        for i in range(len(self.cluster)):
            center = self.data.loc[self.cluster[i]].values.mean(axis=0)
            norm_2 = np.linalg.norm(self.data.loc[self.cluster[i]] - center, 2, axis=1).tolist()
            present_index = self.cluster[i][norm_2.index(min(norm_2))]
            #self.present_skill.append(self.skill[present_index])
            self.present_skill.append(present_index)
        return
    
    def cluseter_agglomerative(self, n_clusters=20, linkage='average', iterate=5):
        """run agglomerative clustering algorithms on the word2vec model
            in order to cluster skill titles
        
           Parameters
           ----------
           n_clusters: `int`, default=20.
               The number of clusters to find in each iterate.
               
           linkage: {“ward”, “complete”, “average”}, default:“average”.
               The linkage criterion determines which distance to use between sets of observation.
           
           iterate: `int`.
               How many times we run agglomerative clustering algorithms.
               After each iterate, we take the biggest cluster as the input of the next cluster process.
        """
        df = pd.DataFrame(self.data.copy())
        skill_list = self.skill.copy()
        for i in tqdm(range(iterate)):
            clustering_agg = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
            result_agg = clustering_agg.fit_predict(df)
            skill_counts = Counter(result_agg)
            top_three = skill_counts.most_common(3)
            for i in range(n_clusters):
                if i != top_three[0][0]:
                    #list_temp = []
                    index_temp = []
                    for j in range(len(result_agg)):
                        if result_agg[j] == i:
                            #list_temp.append(skill_list[j])
                            index_temp.append(self.skill.index(skill_list[j]))
                    self.cluster.append(index_temp)
                    #if show_result:
                        #print(list_temp)
                        #print("  ")
            #if show_result:
                #print("  ")        
                    
            df2 = pd.DataFrame(columns=[np.zeros([self.data.shape[1]])])
            skill_list2 = []
            j = 0
            for i in range(len(result_agg)):
                if result_agg[i] == top_three[0][0]:
                    df2.loc[j] = df.loc[i]
                    skill_list2.append(skill_list[i])
                    j += 1
            df = df2
            skill_list = skill_list2
        
        self.skill_select()
        return 
    
    def print_skill_title(self):
        """print all the skill titles in different clusters
        """
        #index_largest = self.clusters.index(max(self.clusters))
        for i in range(len(self.cluster)):
            #if i != index_largest:
            list_temp = []
            for j in range(len(self.cluster[i])):
                 list_temp.append(self.skill[self.cluster[i][j]])
            #print(self.present_skill[i], list_temp)
            print(i, self.skill[self.present_skill[i]], list_temp)
            print("  ") 
        return 
    
    def sub_clustering(self, n_clusters, index_cluster=None, linkage='complete', n_max=30):
        """run agglomerative clustering algorithms on big clusters
        
           Parameters
           ----------
           index_cluster: `list`, 
                    Contains the index of big clusters which you want to sub-clustering. 
                    None means that search all the clusters and sub_clustering all the big clusters.
               
           linkage: {“ward”, “complete”, “average”}, default:“complete”.
               The linkage criterion determines which distance to use between sets of observation.
               I tried them, and I found that “complete” works best.
           
           n_clusters: `int`.
               The number of sub clusters after clustering.
           
           n_max: `int`.
               If the number of items is more than n_max in a cluster, then we consider it's a big
               cluster, and then we will divide it into sub-clusters.
        """
        cluster_temp = self.cluster.copy()
        #n_clusters=int(len(self.cluster)/n_max*2+1)
        if index_cluster is None:
            for i in range(len(cluster_temp)):
                if len(cluster_temp[i]) > n_max:
                    new_data = self.data.loc[cluster_temp[i]]
                    clustering_agg = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
                    result_agg = clustering_agg.fit_predict(new_data)
                    self.cluster.remove(cluster_temp[i])
                    for k in range(n_clusters):
                        temp_list = []
                        for j in range(len(result_agg)):
                            if result_agg[j] == k:
                                temp_list.append(cluster_temp[i][j])
                        self.cluster.append(temp_list)
        
        else:
            for item in index_cluster:
                new_data = self.data.loc[cluster_temp[item]]
                clustering_agg = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
                result_agg = clustering_agg.fit_predict(new_data)
                self.cluster.remove(cluster_temp[item])
                for k in range(n_clusters):
                    temp_list = []
                    for j in range(len(result_agg)):
                        if result_agg[j] == k:
                            temp_list.append(cluster_temp[item][j])
                    self.cluster.append(temp_list)
            
        self.skill_select()
        return
    
    def sub_clustering_test(self, n_clusters, index_cluster, linkage='aveage'):
        """run agglomerative clustering algorithms a given cluster, to see if the result is satisficing.
            It doesn't change the self.cluster's structure.
        
           Parameters
           ----------
           index_cluster: `int`, 
                    The index of big clusters which you want to sub-clustering. 
               
           linkage: {“ward”, “complete”, “average”}, default:“aveage”.
               The linkage criterion determines which distance to use between sets of observation.
           
           n_clusters: `int`.
               The number of sub clusters after clustering.

        """
        new_data = self.data.loc[self.cluster[index_cluster]]
        clustering_agg = AgglomerativeClustering(n_clusters=n_clusters,linkage=linkage)
        result_agg = clustering_agg.fit_predict(new_data)
        for k in range(n_clusters):
            temp_list = []
            for j in range(len(result_agg)):
                if result_agg[j] == k:
                    temp_list.append(self.skill[self.cluster[index_cluster][j]])
            print(temp_list)
            print("  ")
        return
    
    def merge_clusters(self, index_cluster=None, member_min=2):
        """merge small clusters into a bigger one
        
           Parameters
           ----------
           index_cluster: `list`, 
                    Contains the index of big clusters which you want to merge. 
                    None means that search all the small clusters and merge them.
               
           member_min: `int`, default:“3”.
               The number of members which is smaller than member_min will be considered
               as a small cluster.
        """
        if index_cluster is None:
            merge_list = []
            cluster_temp = self.cluster.copy()
            data_temp = pd.DataFrame(columns=[np.zeros([self.dim])])
            data_index = 0
            for i in self.present_skill:
                data_temp.loc[data_index] = self.data.loc[i]
                data_index += 1
            for i in range(len(cluster_temp)):
                if len(cluster_temp[i]) < member_min:
                    list_temp = data_temp.drop([i]).sub(data_temp.loc[i],axis=1).abs().sum(axis=1).tolist()
                    if len(merge_list) > 0:
                        list_index = -1
                        side = -1
                        for j in range(len(merge_list)):
                            if (i in merge_list[j] or list_temp.index(min(list_temp)) in merge_list[j]):
                                if (i in merge_list[j] and list_temp.index(min(list_temp)) in merge_list[j]):
                                    break
                                elif i in merge_list[j]:
                                    list_index = j
                                    side = 0
                                elif list_temp.index(min(list_temp)) in merge_list[j]:
                                    list_index = j
                                    side = 1
                        if list_index >= 0:
                            if side == 0:
                                merge_list[list_index].append(list_temp.index(min(list_temp)))
                            if side == 1:
                                merge_list[list_index].append(i)
                        else:
                            merge_list.append([i,list_temp.index(min(list_temp))])
                    else:
                        merge_list.append([i,list_temp.index(min(list_temp))])

            #print(merge_list)
            self.merge_clusters(index_cluster=merge_list)
            
        else:
            cluster_temp = self.cluster.copy()
            for i in index_cluster:
                temp_list = [] 
                for j in i:
                    try:
                        self.cluster.remove(cluster_temp[j])
                    except:
                        j=j
                    for k in cluster_temp[j]:
                        temp_list.append(k)
                self.cluster.append(temp_list)
            #for i in index_cluster:
                #for j in i:
                    #self.cluster.remove(cluster_temp[j])
               
        self.skill_select()
        return
            
            
    def fit(self):
        """
        give a manually selected clustering result
        """
        self.cluseter_agglomerative(n_clusters=20, linkage='average', iterate=5)
        self.sub_clustering(n_clusters=3, index_cluster=[79], linkage='complete')
        self.merge_clusters([[0,9,53],[1,83],[46,35,67],[88,23],[6,68]])
        self.merge_clusters([[6,33,52],[17,14]])
        self.sub_clustering(n_clusters=2, index_cluster=[0], linkage='average')
        self.sub_clustering(n_clusters=3, index_cluster=[2], linkage='average')
        self.sub_clustering(n_clusters=3, index_cluster=[85], linkage='average')
        self.sub_clustering(n_clusters=2, index_cluster=[14], linkage='complete')
        self.sub_clustering(n_clusters=2, index_cluster=[16], linkage='average')
        self.sub_clustering(n_clusters=3, index_cluster=[22], linkage='average')
        self.sub_clustering(n_clusters=2, index_cluster=[24], linkage='complete')
        self.sub_clustering(n_clusters=2, index_cluster=[26], linkage='complete')
        self.sub_clustering(n_clusters=3, index_cluster=[28], linkage='ward')
        self.merge_clusters([[6,98,99]])
        self.merge_clusters([[35,80]])
        self.sub_clustering(n_clusters=4, index_cluster=[35], linkage='complete')
        self.merge_clusters([[76,98]])
        self.sub_clustering(n_clusters=3, index_cluster=[35], linkage='complete')
        self.merge_clusters([[39,42]])
        self.sub_clustering(n_clusters=3, index_cluster=[47], linkage='complete')
        self.sub_clustering(n_clusters=3, index_cluster=[51], linkage='average')
        self.merge_clusters([[70,101]])
        self.sub_clustering(n_clusters=3, index_cluster=[51], linkage='complete')
        self.sub_clustering(n_clusters=3, index_cluster=[61], linkage='ward')
        self.merge_clusters()
        return
    
    
    def visualize_in_2d(self):
        """try to visualize the performence of the clustering algorithm by
            ploting orignal data on a 2-d graph with help of Autoencoder
        """
        ae = Autoencoder_test(self.data)
        self.code = ae.encode(n_dimension=2, learning_rate=0.01, training_epochs=10, batch_size=400)
        for i in range(len(self.cluster)):
            list_x = []
            list_y = []
            for j in self.cluster[i]:
                list_x.append(self.code[0][j,0])
                list_y.append(self.code[0][j,1])
            plt.scatter(list_x,list_y)
        plt.show()
        return 
    def partial_visualize_in_2d(self, cluster_index=[5,12,35,44,64,75,81]):
        """visualize the performence of certain clustering resut, need to run self.visualize_in_2d() first.
        
           Input
           ----------
           cluster_index: `list`, 
                    index of clusters that need to be scattered.
        """
        for i in cluster_index:
            list_x = []
            list_y = []
            for j in self.cluster[i]:
                list_x.append(self.code[0][j,0])
                list_y.append(self.code[0][j,1])
            plt.scatter(list_x,list_y, label=self.skill[self.present_skill[i]])
        plt.legend()
        plt.show()
        return
    
class Autoencoder_test:
    
    def __init__(self, data):
        self.data = data
        self.n_input = data.shape[1]
        
    def encode(self, n_dimension=2, learning_rate=0.01, training_epochs=10, batch_size=400):
        """implement autoencoder to get an encoder with smaller dimension 
        
           Parameters
           ----------
           n_dimension: `int`.
               output encoder's dimension, usually equals to 2,
               in order to be demonstrated on 2-D graph.
               
           learning_rate: `float`.
               learning rate of neural network.
           
           training_epochs: `int`.
               number of times run optimizer over whole data.
               
           batch_size: `int`.
               number of items that be ran every time.
               
           Output:
           ----------
           self.X_test: `np.array`.
               shape: (data[0],n_dimension)
               compressed data, on which we can test clustering algorithms.
               
           self.X_cost: `float`. 
               to see if this encoder loss much information.
        """
        X = tf.placeholder(tf.float32,[None, self.n_input])
        tf.set_random_seed(50)
        
        
        n_hidden_layer1 = int(math.pow(2, int(2*math.log(self.n_input,2)/3+math.log(n_dimension,2)/3)))
        n_hidden_layer2 = int(math.pow(2, int(math.log(self.n_input,2)/3+2*math.log(n_dimension,2)/3)))
        n_hidden_layer3 = n_dimension
        
        weights = {
            'encoder_w1':tf.Variable(tf.random_normal([self.n_input, n_hidden_layer1])),
            'encoder_w2':tf.Variable(tf.random_normal([n_hidden_layer1, n_hidden_layer2])),
            'encoder_w3':tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer3])),
        
            'decoder_w1':tf.Variable(tf.random_normal([n_hidden_layer3, n_hidden_layer2])),
            'decoder_w2':tf.Variable(tf.random_normal([n_hidden_layer2, n_hidden_layer1])),
            'decoder_w3':tf.Variable(tf.random_normal([n_hidden_layer1, self.n_input])),
         }
        
        biases = {
            'encoder_b1':tf.Variable(tf.random_normal([n_hidden_layer1])),
            'encoder_b2':tf.Variable(tf.random_normal([n_hidden_layer2])),
            'encoder_b3':tf.Variable(tf.random_normal([n_hidden_layer3])),
        
            'decoder_b1':tf.Variable(tf.random_normal([n_hidden_layer2])),
            'decoder_b2':tf.Variable(tf.random_normal([n_hidden_layer1])),
            'decoder_b3':tf.Variable(tf.random_normal([self.n_input])),
         }
        
        
        def encoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w1']), biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_w2']), biases['encoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_w3']), biases['encoder_b3']))
    
            return layer_3

        def decoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w1']), biases['decoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_w2']), biases['decoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_w3']), biases['decoder_b3']))
    
            return layer_3
        
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)

        y_pred = decoder_op
        y_true = X

        cost = tf.reduce_mean(tf.pow(y_pred - y_true, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        
        
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            n_batch = int(self.data.shape[0]/batch_size)
            for epoch in tqdm(range(training_epochs)):
                for batch_idx in range(n_batch):
                    start = batch_idx * batch_size
                    stop = start + batch_size
                    _, encoder_result = sess.run([optimizer, encoder_op], feed_dict={X: self.data[start:stop]})
            self.X_test = sess.run(encoder_op, feed_dict={X:self.data})
            self.X_cost = sess.run(cost, feed_dict={X:self.data})
            
        return self.X_test, self.X_cost
 