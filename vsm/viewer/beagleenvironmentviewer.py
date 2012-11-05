import vsm.viewer as vw
import similarity


class BeagleEnvironmentViewer(vw.Viewer):

    def __init__(self,
                 corpus=None,
                 matrix=None,
                 tok_name='sentences'):
        
        super(BeagleEnvironmentViewer, self).__init__(corpus=corpus,
                                                      matrix=matrix,
                                                      tok_name=tok_name)
        
        self._term_norms = None
        
        

    @property
    def term_norms(self):

        if self._term_norms is None:

            self._term_norms = similarity.row_norms(self.matrix)

        return self._term_norms



    def similar_terms(self, term, filter_nan=True, rem_masked=True):

        return vw.similar_terms(self.corpus,
                                self.matrix,
                                term,
                                norms=self.term_norms,
                                filter_nan=filter_nan,
                                rem_masked=rem_masked)



    def mean_similar_terms(self, query, filter_nan=True, rem_masked=True):

        return vw.mean_similar_terms(self.corpus,
                                     self.matrix,
                                     query,
                                     norms=self.term_norms,
                                     filter_nan=filter_nan,
                                     rem_masked=rem_masked)



    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus, self.matrix, term_list)
