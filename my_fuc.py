def series_to_df(series,name_str):#series:temp_advice.groupby('药品通用名称').size()
  df = series.to_frame()
  df.reset_index(inplace=True)
  df = df.rename(columns={0 : name_str})
  df.sort_values(by='count',ascending=False,inplace=True)
  return df