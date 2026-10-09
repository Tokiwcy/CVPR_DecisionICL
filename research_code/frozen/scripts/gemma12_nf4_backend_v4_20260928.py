"""Process-local strict recovery for an empty official assistant parse.
The frozen v2 backend and official parser files are not modified on disk.
"""
import gemma12_nf4_backend_v2_20260926 as v2

spec=v2.spec
OFFICIAL_FINAL_TEXT=spec.final_text
POLICY='Only exact dict {role: assistant} with absent content maps to empty text; exact-label prediction maps it to INVALID. Preserve parsed object and decoded raw. All other schemas use the unchanged official final_text and still raise.'

def final_text_role_only_invalid(parsed):
    if type(parsed) is dict and parsed == {'role':'assistant'}:
        return ''
    return OFFICIAL_FINAL_TEXT(parsed)

final_text_role_only_invalid._gemma12_v7_role_only_invalid=True

def install():
    spec.final_text=final_text_role_only_invalid
    v2.CONTRACT=dict(v2.CONTRACT,output_parsing=POLICY)
    return v2.Gemma12Backend

def self_check():
    assert final_text_role_only_invalid({'role':'assistant'})==''
    assert final_text_role_only_invalid({'role':'assistant','content':''})==''
    assert final_text_role_only_invalid({'role':'assistant','content':'cat'})==OFFICIAL_FINAL_TEXT({'role':'assistant','content':'cat'})
    assert final_text_role_only_invalid({'role':'assistant','content':[{'type':'text','text':'dog'}]})==OFFICIAL_FINAL_TEXT({'role':'assistant','content':[{'type':'text','text':'dog'}]})
    for bad in ({'role':'user'}, {'role':'assistant','content':None}, {'role':'assistant','extra':'x'}, []):
        try:
            final_text_role_only_invalid(bad)
        except (ValueError,TypeError):
            pass
        else:
            raise AssertionError(f'unknown schema was accepted: {bad!r}')
    return {'role_only_missing_content':'empty_text_then_strict_invalid','valid_content':'delegates byte-for-byte to frozen official extractor','unknown_schema':'still rejected','raw_persistence':'forward returns decoded raw and parsed object after recognized empty parse'}
