import unittest2 as unittest
import tempfile
import shutil
import os
import json
import filecmp

from vsm.corpus.mapcorpus import *



class TestMapcorpus(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestMapcorpus, self).__init__(*args, **kwargs)
        self.maxDiff = None


    def setup_corpus_tree(self, corpus_dir):

        os.makedirs(os.path.join(corpus_dir, 'documents'))
        os.makedirs(os.path.join(corpus_dir, 'metadata'))

        corp_meta_file = os.path.join(corpus_dir, 'metadata', 
                                      'corpus.metadata.json')

        doc_file_0 = os.path.join(corpus_dir, 'documents', '0.json')
        doc_file_1 = os.path.join(corpus_dir, 'documents', '1.json')

        doc_meta_file_0 = os.path.join(corpus_dir, 'metadata', 
                                       '0.metadata.json')
        doc_meta_file_1 = os.path.join(corpus_dir, 'metadata', 
                                       '1.metadata.json')
        
        with open(doc_file_0, 'w') as f:
            json.dump(['form without content 0'], f)
        with open(doc_file_1, 'w') as f:
            json.dump(['form without content 1', 
                       'form without content 2'], f)
        with open(doc_meta_file_0, 'w') as f:
            json.dump([ { 'user_defined': {},
                          'file_idx': 0,
                          'local_idx': 0,
                          'global_idx': 0 } ], f, sort_keys=True)
        with open(doc_meta_file_1, 'w') as f:
            json.dump([ { 'user_defined': {},
                          'file_idx': 1,
                          'local_idx': 0,
                          'global_idx': 1 },
                        { 'user_defined': {},
                          'file_idx': 1,
                          'local_idx': 1,
                          'global_idx': 2 } ], f, sort_keys=True)
        with open(corp_meta_file, 'w') as f:
            json.dump({ 'user_defined': {},
                        'files': [ [ '0.json', '0.metadata.json' ], 
                                   [ '1.json', '1.metadata.json' ] ],
                        'doc_paths': [ [0, 0], [1, 0], [1, 1] ] }, 
                      f, sort_keys=True)


    def setup_make_two_corpus_dirs(self, corpus_dir_1='corpus_dir_1', 
                                   corpus_dir_2='corpus_dir_2'):

        self.tempdir = tempfile.mkdtemp()

        cd1 = os.path.join(self.tempdir, corpus_dir_1)
        cd2 = os.path.join(self.tempdir, corpus_dir_2)

        os.makedirs(cd1)
        os.makedirs(cd2)

        return cd1, cd2
        

    def filecmp_corpus_dirs(self, corpus_dir_1, corpus_dir_2,
                            files=[ 'documents/0.json', 
                                    'documents/1.json', 
                                    'metadata/0.metadata.json', 
                                    'metadata/1.metadata.json', 
                                    'metadata/corpus.metadata.json' ]):

        match, mismatch, errors = filecmp.cmpfiles(corpus_dir_1, corpus_dir_2, 
                                                   files)

        return not (mismatch or errors)


    def setup_src_files(self, corpus_dir):

        doc_src_files = [ os.path.join(corpus_dir, '0.json'),
                          os.path.join(corpus_dir, '1.json') ]

        with open(doc_src_files[0], 'w') as f:
            json.dump(['form without content 0'], f)
        with open(doc_src_files[1], 'w') as f:
            json.dump(['form without content 1', 
                       'form without content 2'], f)

        return doc_src_files


    def setUp(self):

        self.archive = { 'documents': 
                         { '0.json': 
                           [ 'form without content 0' ], 
                           '1.json': 
                           [ 'form without content 1',
                             'form without content 2' ] },
                         'metadata': 
                         { 'corpus.metadata.json': 
                           { 'user_defined': {},
                             'files':
                             [ [ '0.json', '0.metadata.json' ], 
                               [ '1.json', '1.metadata.json' ] ],
                             'doc_paths': 
                             [ [0, 0], [1, 0], [1, 1] ] },
                           '0.metadata.json': 
                           [ { 'user_defined': {},
                               'file_idx': 0,
                               'local_idx': 0,
                               'global_idx': 0 } ],
                           '1.metadata.json':
                           [ { 'user_defined': {},
                               'file_idx': 1,
                               'local_idx': 0,
                               'global_idx': 1 },
                             { 'user_defined': {},
                               'file_idx': 1,
                               'local_idx': 1,
                               'global_idx': 2 }] } }
        

    def test_archive_to_file_tree(self):

        cd1, cd2 = self.setup_make_two_corpus_dirs()
        self.setup_corpus_tree(cd1)
        archive_to_file_tree(self.archive, cd2)
    
        self.assertTrue(self.filecmp_corpus_dirs(cd1, cd2))


    def test_file_tree_to_archive(self):

        self.tempdir = tempfile.mkdtemp()
        self.setup_corpus_tree(self.tempdir)

        self.assertEquals(self.archive, file_tree_to_archive(self.tempdir))


    # def test_doc_paths(self):

    #     out = [ [0, 0], [1, 0], [1, 1] ]

    #     self.assertEquals(out, doc_paths(archive=self.archive))
        
    #     self.tempdir = tempfile.mkdtemp()
    #     self.setup_corpus_tree(self.tempdir)

    #     self.assertEquals(out, doc_paths(corpus_dir=self.tempdir))


    def test_autogen_corpus(self):

        cd1, cd2 = self.setup_make_two_corpus_dirs()
        self.setup_corpus_tree(cd1)
        src_files = self.setup_src_files(self.tempdir)
        autogen_corpus(src_files, corpus_dir=cd2)

        archive = autogen_corpus(src_files, archive=True)
        
        self.assertTrue(self.filecmp_corpus_dirs(cd1, cd2))
        self.assertEquals(self.archive, archive)


    def test_map_corpus(self):

        fn = lambda *args: args

        cd1, cd2 = self.setup_make_two_corpus_dirs()
        self.setup_corpus_tree(cd1)
        map_corpus(fn, corpus_dir=cd2, src_dir=cd1, global_idx=True)

        archive = map_corpus(fn, archive=self.archive, global_idx=True)
        
        self.assertTrue(self.filecmp_corpus_dirs(cd1, cd2))
        self.assertEquals(self.archive, archive)


    def tearDown(self):

        if hasattr(self, 'tempdir'):
            shutil.rmtree(self.tempdir)
            del self.tempdir


suite = unittest.TestLoader().loadTestsFromTestCase(TestMapcorpus)
unittest.TextTestRunner(verbosity=2).run(suite)
