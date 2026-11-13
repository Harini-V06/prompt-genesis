"""artifacts/model_card.json for the app's About page. Every number is READ from a report file
written by an evaluation script; nothing is typed by hand."""
import json
R = lambda p: json.load(open(f"reports/{p}"))
pct = lambda x: f"{100 * x:.1f}%"
ci = lambda c: f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"
logo, lopo, band = R("guard_logo_stylo_lm_essay_fixed.json")["pooled"], R("guard_lopo_stylo_lm_essay_fixed.json")["pooled"], R("guard_band.json")
det, ver, dc = R("detective_stylo_lm_tuned.json"), R("verifier_validity_summary.json"), R("deployed_checks.json")
g = lambda d: d.get("guard", d)
card = {
  "AI vs human, AI model never seen in training (6 models held out in turn)": f"{pct(g(logo)['acc'])} accuracy {ci(g(logo)['acc_ci'])}, AUC {g(logo)['roc_auc']:.3f}",
  "AI vs human, essay topic never seen (15 topics held out in turn)": f"{pct(g(lopo)['acc'])} accuracy {ci(g(lopo)['acc_ci'])}",
  "Calibration error (ECE, 0 = perfect)": f"{g(logo)['ece']:.3f}",
  "'Insufficient evidence' band": f"P(AI) between {band['abstain_if_p_between'][0]:.2f} and {band['abstain_if_p_between'][1]:.2f}: answers {pct(band['coverage'])} of essays at {pct(band['acc_answered'])} accuracy",
  "Prompt settings (Detective, unseen topics)": {k: f"{pct(v['acc'])} (confident on {pct(v['coverage_at_threshold'])} at {pct(v['acc_at_threshold'])})" for k, v in det.items()},
  "Verifier: true prompt beats a wrong prompt": f"{pct(ver['match']['share_true_higher'])} of {ver['match']['n']} essays (Match {ver['match']['mean_true']:.2f} vs {ver['match']['mean_wrong']:.2f}; p = {ver['match']['p_bh']:.1e})",
  "Non-native student essays wrongly flagged (TOEFL)": f"{pct(dc['toefl_nonnative']['flagged'])} {ci(dc['toefl_nonnative']['flagged_ci'])}; confidently {pct(dc['toefl_nonnative']['flagged_confident'])} (n={dc['toefl_nonnative']['n']})",
  "Adult professional writing wrongly flagged (OUT OF SCOPE)": f"{pct(dc['adult_prose_raid']['flagged'])} {ci(dc['adult_prose_raid']['flagged_ci'])}; confidently {pct(dc['adult_prose_raid']['flagged_confident'])} (n={dc['adult_prose_raid']['n']})",
}
json.dump(card, open("artifacts/model_card.json", "w"), indent=1)
print(json.dumps(card, indent=1))
