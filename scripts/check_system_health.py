import sys
from integrations.firebase_client import FirebaseConnector


KEY_FILE = "projektsystemekspertowy-firebase-adminsdk-fbsvc-c6d7fc4e7f.json"

def main():
    print("\n🔍 STARTING SYSTEM HEALTH CHECK...\n")
    

    try:
        connector = FirebaseConnector(KEY_FILE)
    except Exception as e:
        print(f"❌ CRITICAL: Failed to initialize Firebase client.")
        print(f"   Reason: {e}")
        sys.exit(1)


    print("   Running connectivity tests...")
    results = connector.check_connection()


    print(f"\n📊 DIAGNOSTIC REPORT")
    print(f"   Target Bucket: {results['bucket_name']}")
    print("   ----------------------------------------")

    checks = [
        ("Internet Connection", results["internet"]),
        ("Firestore Access   ", results["firestore"]),
        ("Storage Access     ", results["storage"])
    ]

    all_passed = True
    for name, passed in checks:
        icon = "✅" if passed else "❌"
        status_text = "OK" if passed else "FAIL"
        print(f"   {icon} {name}: {status_text}")
        if not passed:
            all_passed = False

    print("\n-------------------------------------------")
    if all_passed:
        print("🚀 SYSTEM STATUS: GREEN. Ready for operations.\n")
    else:
        print("⚠️  SYSTEM STATUS: RED. Check logs above.\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
