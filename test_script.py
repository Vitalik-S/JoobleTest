from jobstand import JobFeaturePreprocess, ZScoreScaler


jfp = JobFeaturePreprocess('train.tsv', 'test.tsv', ZScoreScaler())
preprocessed_data = jfp.preprocess_data()
preprocessed_data.to_csv('test_proc.tsv', sep='\t', index=False)