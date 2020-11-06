# interviewbit-imp-question
*****************************************************************************************
    bool compare(Interval A,Interval B) 
    {
        return A.start < B.start;
    }
    vector<Interval> Solution::merge(vector<Interval> &A) {
    int n=A.size(); 
    sort(A.begin(),A.end(),compare); 
    //for(int i=0;i<A.size();i++) 
        //cout<<A.start<<" ";
    vector<Interval> ans;
    Interval b=Interval(0,0); 
    for(int i=0;i<A.size();i++) 
    {
        if(i==0) 
        {
            b.start=A[i].start;
            b.end=A[i].end;
        } 
        else 
        {
            if(A[i].start <= b.end) 
                b.end=max(b.end,A[i].end);  // change here for overlapping with same value 
            else 
            {
                ans.push_back(b);
                b=Interval(A[i].start,A[i].end);
            }  
        }
    } 
    ans.push_back(b);
    return ans;
    }
*****************************************************************************************
      vector<int> Solution::nextPermutation(vector<int> &A) {
          next_permutation(A.begin(),A.end());
          return A;
      }

*****************************************************************************************
      class Solution {
      public:
          vector<int> repeatedNumber(const vector<int> &V) {
             long long sum = 0;
             long long squareSum = 0;
             long long temp;
             for (int i = 0; i < V.size(); i++) {
                 temp = V[i];
                 sum += temp;
                 sum -= (i + 1);
                 squareSum += (temp * temp);
                 squareSum -= ((long long)(i + 1) * (long long)(i + 1));
             }
             // sum = A - B
             // squareSum = A^2 - B^2 = (A - B)(A + B)
             // squareSum / sum = A + B
             squareSum /= sum;

             // Now we have A + B and A - B. Lets figure out A and B now. 
             int A = (int) ((sum + squareSum) / 2);
             int B = squareSum - A;

             vector<int> ret;
             ret.push_back(A);
             ret.push_back(B);
             return ret;
          }
      };

*****************************************************************************************
      int Solution::repeatedNumber(const vector<int> &A) {
          const int k = 3;
          int n = A.size();
          vector<int> cnt(k, 0), candidate(k, -1);
          for (int i = 0; i < n; i++) {
              bool updated = false;
              for (int j = 0; j < k; j++) {
                  if (candidate[j] == A[i]) {
                      cnt[j]++, updated = true;
                      break;
                  }
              }
              if (updated) continue;
              for (int j = 0; j < k; j++) {
                  if (cnt[j] == 0) {
                      cnt[j]++, updated = true;
                      candidate[j] = A[i];
                      break;
                  }
              }
              if (updated) continue;
              for (int j = 0; j < k; j++)
                  cnt[j]--;
          }
          cnt.assign(k, 0);
          for (int i = 0; i < n; i++) {
              for (int j = 0; j < k; j++) {
                  if (candidate[j] == A[i]) {
                      cnt[j]++;
                      break;
                  }
              }
          }
          for (int j = 0; j < k; j++)
              if (cnt[j] > n / k) return candidate[j];
          return -1;
      }
*****************************************************************************************
       vector<Interval> Solution::insert(vector<Interval> &intervals, Interval newInterval) {
         vector<Interval>v;
          int n=intervals.size();

          for(i=0;i<(n-1);i++){
              if(intervals[i].end>=intervals[i+1].start){

              }
          }
      }
******************************************************************************************
      vector<int> Solution::flip(string A) {
          int meh=0,m=INT_MIN,l=1,r=1,l1=1,count=0;
          for(int i=0;i<A.size();i++){
              if(A[i]=='0'){
                  meh++;
                  count++;
                  if(m<meh){
                      m=meh;
                      r=i+1;
                      l=l1;
                  }
              }else{
                  meh--;
                  if(meh<0){
                      meh=0;
                      l1=i+2;
                  }
              }
          }
          if(count==0)
              return {};
          return {l,r};
      }

****************************************************************************************
      void helper(vector<int>&A,int i,int j,vector<vector<int>>&out,int n){
          if(i==n){
              out.push_back(A);
              return;
          }
          for(j=i;j<n;j++){
              swap(A[i],A[j]);
              helper(A,i+1,j+1,out,n);
              swap(A[i],A[j]);
          }
          return;
      }

      vector<vector<int> > Solution::permute(vector<int> &A) {
          vector<vector<int>>out;
          helper(A,0,0,out,A.size());
          return out;
      }

*****************************************************************************************
vector<vector<int> > Solution::subsetsWithDup(vector<int> &A) {
    set<vector<int>>v;
    sort(A.begin(),A.end());
    vector<int>subset;
    ans(v,subset,A,0);
    vector<vector<int>>a(v.begin(),v.end());
    return a;
}
*****************************************************************************************
solve(vector<int> &A,int n, vector<int> &temp,set<vector<int>> &ans,int index=0){
    if(index == n){
        ans.insert(temp);
        return ;
    }
    if(index >n)
        return ;
    for(int j=)
}

vector<vector<int> > Solution::permute(vector<int> &A) {
    int n=A.size();
    set<vector<int>>ans;
    solve(A,n,temp,ans);
    vector<vector<int>>res;
    for(auto x: ans)
        res.push_back(x);
    return res;
}

*****************************************************************************************
void generate(int A,vector<string> &vec, string s="", int unbalance=0){
    if(s.length() == 2*A and not unbalance){
        vec.push_back(s);
        return;
    }
    if(s.length() >2*A) return;
    if(unbalance < 0 or unbalance > A)return;
    generate(A,vec,s + "(", unbalance + 1);
    generate(A,vec,s + ")", unbalance - 1);
}

vector<string> Solution::generateParenthesis(int A) {
    vector<string>vec;
    generate(A,vec);
    return vec;
}
*****************************************************************************************
void solve(string &A,vector<string> &vec,string s="",int index=0){
    if(index==A.length()){
        vec.push_back(s);
    }
    if(s[i]=="2"){
        
    }
}
vector<string> Solution::letterCombinations(string A) {
    vector<string>vec;
    solve(A,vec);
    return vec;
    
}

******************************************************************************************
void generate(vector<int> &vec, int A, int curVal, bool turn = true) {
    if (not A) {
        vec.push_back(curVal);
        return;
    }
    if (turn) {
        generate(vec, A-1, (curVal<<1)+0, turn);
        generate(vec, A-1, (curVal<<1)+1, !turn);
    } else {
        generate(vec, A-1, (curVal<<1)+1, !turn);
        generate(vec, A-1, (curVal<<1)+0, turn);
    }
}

vector<int> Solution::grayCode(int A) {
    vector<int> vec;
    generate(vec, A, 0);
    return vec;
}

****************************************************************************************
void backtrack(vector &A,int index,vector<vector> &ans,vector &temp){
    ans.push_back(temp);
    for(int i=index;i<A.size();i++){
        temp.push_back(A[i]);
        backtrack(A,i+1,ans,temp);
        temp.pop_back();
    }
}
vector<vector<int> > Solution::subsets(vector<int> &A) {
    vector<vector> ans;
    vector temp;
    sort(A.begin(),A.end());
    backtrack(A,0,ans,temp);
    return ans;
}

*****************************************************************************************
bool isvalid(vector<vector<char>> &b, int r, int c, char k) {

    for(int i = 0; i<9; ++i){
        if(b[r][i] == k) return false;
    }

    for(int i = 0; i<9; ++i){
        if(b[i][c] == k) return false;
    }

    int blockx = r/3;
    int blocky = c/3;
    for(int i = 3*blockx; i< 3*blockx+3; ++i){
        for(int j = 3*blocky; j<3*blocky+3; ++j){
            if(b[i][j] == k) return false;
        }
    }
    return true;
}

bool solver(vector<vector<char>> &b){
    for(int i = 0; i < 9; ++i){
        for(int j = 0; j<9; ++j){
            if(b[i][j] == '.'){
                for(int k = 1; k < 10; ++k){
                   if(isvalid(b, i, j, (char)('0'+k))){
                        b[i][j] = (char)('0' + k);
                            
                        if(solver(b)) return true;
                        b[i][j] = '.';
                   }
                }
                return false;
            }
        }
    }

    return true;
}
void Solution::solveSudoku(vector<vector<char> > &A) {
    solver(A);
}

*****************************************************************************************
bool isValid(vector<string> &board, string &s, int row, int col, int n){
    if(s[col]=='Q') return false; // for current row

    int j=col, k=1;
    for(int i=row-1; i>=0; i--){
        if(j+k<n && board[i][j+k]=='Q') // upper right diagonal
            return false;
        if(j-k>=0 && board[i][j-k]=='Q') // upper left diagonal
            return false;
        k++;
    }
    return true;
}

void backtracker(vector<vector<string>> &soln, vector<string> &board, int row, int n, string &s){
    if(row==n){
        soln.push_back(board);
        return;
    }
    for(int col=0; col<n; col++){
        if(isValid(board, s, row, col, n)){
            board[row][col] = 'Q'; s[col] = 'Q';
            backtracker(soln, board, row+1, n, s);
            board[row][col] = '.'; s[col] = '.';
        }
    }
}

vector<vector<string> > Solution::solveNQueens(int A) {
    vector<vector<string>> soln;
    vector<string> board(A, string(A,'.'));
    string s(A, '.');
    backtracker(soln, board, 0, A, s);
    return soln;
}

*****************************************************************************************
int Solution::searchInsert(vector<int> &A, int B) {
    int n = A.size();
    if(B>A[n-1])
        return n;
    int start=0;
    int end = n-1;
    int ans;
    while(start<=end)
    {
        int mid = start + (end-start)/2;
        if(A[mid]==B)
            return mid;
        if(A[mid]>B)
        {
            ans = mid;
            end = mid-1;
        }
        else
            start = mid+1;
    }
    return ans;
    
}

*****************************************************************************************
void helper( vector<int> A,int B, int start , int end , int &mn)
{
    while(start<=end)
    {
        int mid = start + (end - start)/2;
        if(A[mid]==B)
            mn = min(mn,mid);
        if(A[mid] < B)
            start = mid+1;
        else 
            end = mid-1;
    }
}
void helper1( vector<int> A,int B, int start , int end , int &mx)
{
    while(start<=end)
    {
        int mid = start + (end - start)/2;
        if(A[mid]==B)
            mx = max(mx,mid);
        if(A[mid] <= B)
            start = mid+1;
        else 
            end = mid-1;
    }
}

vector<int> Solution::searchRange(const vector<int> &A, int B) {
    int n = A.size();
    int mn =INT_MAX,mx=INT_MIN;
    helper(A,B,0,n-1,mn);
    helper1(A,B,0,n-1,mx);
    if(mn == INT_MAX)
        mn = -1;
    if(mx == INT_MIN)
        mx = -1;
    return vector<int> {mn,mx};
}
******************************************************************************************
****************************************************************************************
*****************************************************************************************
*****************************************************************************************
double Solution::findMedianSortedArrays(const vector<int> &A, const vector<int> &B) {
        int arr[A.size()],brr[B.size()];
        for(int i=0;i<A.size();i++)arr[i]=A[i];
        for(int i=0;i<B.size();i++)brr[i]=B[i];
        
        int l1=A.size();
        int l2=B.size();
        if(l1>l2){
            int end=l1-1;
            int start=0;
            for(int i=0;i<l2;i++){
                if(brr[end]<arr[start]){
                    swap(brr[end],arr[start]);
                    end--;
                    start++;
                }else{
                  break;  
                }
            }
            sort(arr,arr+l1);
            return arr[(l1+l2)/2];
        }else{
            
        }
    
}

*****************************************************************************************
int Solution::singleNumber(const vector<int> &A) {
    int ans=0,len=A.size(),even,odd,p=1;
    
    for(int i=0;i<32;i++){
        even=0;
        odd=0;
        for(int j=0;j<len;j++){
            if(p&A[j]==0){
                even++;
            }else odd++;
        }
        if(odd%3!=0)
            ans+=p;
        p*=2;
    }
    return ans;
    
}
*****************************************************************************************
******************************************************************************************
****************************************************************************************
int solve(TreeNode* A, int &res){
     if(A==NULL)return 0;
    int l=solve(A->left,res);
    int r=solve(A->right,res);
    
    int temp=max(max(l,r)+A->val,A->val);
    
    int ans=max(temp,l+r+A->val);
    res = max(res,ans);
    
    return temp;
    
 }
 
 
int Solution::maxPathSum(TreeNode* A) {
    int res=INT_MIN;
    solve(A,res);
    return res;
}
*****************************************************************************************
*****************************************************************************************
int Solution::solve(vector<int> &A, int B) {
    int n=A.size();
    int pre[n+4]={0},suf[n+4]={0};
    
    int i,j,k;
    
    for(i=1;i<=n;i++){
        pre[i]=min(pre[i-1],A[i-1]);
    }
    for(i=(n-1);i>=0;i--){
        suf[i+1]=max(suf[i+2],A[i]);
    }
    int diff[n+4]={0};
    for(i=1;i<=n;i++){
        diff[i]=(suf[i]-pre[i]);
    }
    sort(diff+1,diff+n+1);
    int sum=0;
    for(i=n;i>=(n-B+1);i--){
        sum+=diff[i];
    }
    
    return sum;
    
}

*****************************************************************************************
vector<vector<int> > Solution::solve(int A, vector<vector<int> > &B) {
    if(A==0)    
        return B;
    int r=B.size();
    int c=B[0].size();
    
    vector<vector<int>>C=B;
    
    for(int m=1;m<=A;m++){
        for(int i=0;i<r;i++){
            for(int j=0;j<c;j++){
                int neigh_x[] = {0,1,0,-1};
                int neigh_y[] = {1,0,-1,0}
                
                int max_neigh = B[i][j];
                for(int t=0;t<4;t++){
                    int neigh_pos_x=i+neigh_x[t];
                    int neigh_pos_y=j+neigh_y[t];
                    if(neigh_pos_x>=0 and neigh_pos_x<r and neigh_pos_y>=0 and neigh_pos_y<c){
                        max_neigh = max(max_neigh, B[neigh_pos_x][neigh_pos_y]);
                    }
                }
                
                C[i][j]=max_neigh;
                
             }
        }
        B=C;
    }
    
    return C;
    
}

*****************************************************************************************
int Solution::maxProfit(const vector<int> &A) {
    if(A.size()==0 or A.size()==1)return 0;
    int fb=INT_MIN, sb=INT_MIN, fs=INT_MIN, ss=INT_MIN;
    for(int i=0;i<A.size();i++){
        fb=max(fb,-A[i]);
        fs=max(fs,fb+A[i]);
        sb=max(sb,fs-A[i]);
        ss=max(ss,sb+A[i]);
        
    }
    return ss;
}

******************************************************************************************
int Solution::isMatch(const string A, const string B) {
    int n=A.length();
    int m=B.length();
    bool dp[n+2][m+2];
    memset(dp,false,sizeof(dp));
    
    dp[0][0]=true;
    for(int i=1;i<=n;i++)
        dp[i][0]=false;
    for(int i=1;i<=m;i++){
        dp[0][i]=false;
        if(i>=2 and B[i-1]=='*')
            dp[0][i]=dp[0][i-2];
    }
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++){
            if(A[i-1]==B[j-1] || B[j-1]=='.')
                dp[i][j]=dp[i-1][j-1];
            else if(B[j-1]=='*'){
                if(j>=2 and (A[i-1]==B[j-2] || B[j-2]=='.'))
                    dp[i][j]=dp[i-1][j]|dp[i][j-2];
                else if(j>=2)
                    dp[i][j]=dp[i][j-2];
            }
        }
    }
    return dp[n][m];
    
}

****************************************************************************************
int compute(vector<int> &A, int l, int r, bool turn, vector<vector<int> > &dp) {
    if(l>r)                     return 0;
    if(l==r)                    return turn ? A[l] : 0;
    if(dp[l][r] != -1)          return dp[l][r];
    int left = compute(A, l+1, r, !turn, dp);
    int right = compute(A, l, r-1, !turn, dp);
    if(turn)
        dp[l][r] = max(left+A[l], right+A[r]);
    else
        dp[l][r] = min(left, right);
    return dp[l][r];
}
int Solution::maxcoin(vector<int> &A) {
    int len = A.size();
    vector<vector<int> > dp(len, vector<int>(len, -1));
    dp[0][len-1] = compute(A, 0, len-1, true, dp);
    return dp[0][len-1];
}
*****************************************************************************************
int dp[701][701];
int func(int i,int j,string A,string B){
    if(i>=A.size() || j>=B.size()){
        if(j==B.size())return 1;
        return 0;
        
    }
    if(dp[i][j]!=-1)return dp[i][j];
    dp[i][j]=0;
    if(A[i]==B[j])
        dp[i][j]+=func(i+1,j+1,A,B);
    dp[i][j]+=func(i+1,j,A,B);
    return dp[i][j];
        
}

int Solution::numDistinct(string A, string B) {
    memset(dp,-1,sizeof dp);
    return func(0,0,A,B);
}

******************************************************************************************
void wordbreak(int index, string &s, string temp,vector<string> &res, unordered_map<string,int> &mp){
    if(index == s.size()){
        temp.pop_back();
        res.push_back(temp);
        return;
    }
    string sub="";
    for(int i=index;i<s.size();i++){
        sub+=s[i];
        if(mp.find(sub)!=mp.end()){
            
        }
    }
}

vector<string> Solution::wordBreak(string A, vector<string> &B) {
    vector<string>res;
    unordered_map<string,int>mp;
    for(int i=0;i<B.size();i++)
        mp[B[i]]++;
    wordbreak(0,A,"",res,mp);
    return res;
}

****************************************************************************************
bool dfs(string A, int ind, vector<int> &dp,unordered_map<string, int> &dict,int ms){
    if(A.size() == ind)return true;
    if(dp[ind]!=-1)return (bool)dp[ind];
    bool flag=false;
    string s="";
    for(int i=ind;i<A.size();i++){
        s+=A[i];
        if(s.size()>ms)break;
        if(dict[s])flag|=dfs(A,i+1,dp,dict,ms);
    }
    return dp[ind]=flag;
}

int Solution::wordBreak(string A, vector<string> &B) {
    unordered_map<string,int>dict;
    vector<int>dp(A.size()+1,-1);
    int ms=0;
    for(auto x: B)ms=max(ms,(int)x.size()),dict[x]++;
    return (int)dfs(A,0,dp,dict,ms);
}

*****************************************************************************************
int Solution::solve(const vector<int> &A, const vector<int> &B, const vector<int> &C) {
    int t = *max_element(A.begin(), A.end());
    int dp[t+1];
    memset(dp,-1,sizeof(dp));
    dp[0]=0;
    int k=0;
    for(auto i:B){
        for(int j=i;j<=t;j++){
            if(dp[j-i]!=-1)
                dp[j]=dp[j]==-1 ? dp[j-i]+C[k]:min(dp[j],dp[j-i]+C[k]);
        }
        k++;
    }
    int ans=0;
    for(auto i:A)ans+=dp[i];
    return ans;
    
}

******************************************************************************************
int solve(vector<vector<int>> &A,int s,int e, int n,int m,vector<vector<int>> &dp){
    if((s+1)==n and (e+1)==m)   
        return A[s][e];
    if(dp[s][e]!=-1)
        return dp[s][e];
    if(s+1==n)
        return dp[s][e]=min(A[s][e],A[s][e]+solve(A,s,e+1,n,m,dp));
    else if(e+1==m)
        return dp[s][e]=min(A[s][e],A[s][e]+solve(A,s+1,e,n,m,dp));
    else
        return dp[s][e]=max(min(A[s][e],A[s][e]+solve(A,s+1,e,n,m,dp)),
                            min(A[s][e],A[s][e]+solve(A,s,e+1,n,m,dp)));
}

int Solution::calculateMinimumHP(vector<vector<int> > &A) {
    int n,m;
    n=A.size();
    m=A[0].size();
    vector<vector<int>>dp(n,vector(m,-1));
    int ans=solve(A,0,0,n,m,dp);
    if(ans>0)
        return 1;
    return abs(ans-1);
}

****************************************************************************************
int MOD=1000000007;

int func(vector<int>& arr,int sum,int k,vector<vector<int>>& t){
    if(sum==0 and k==0)
        return 1;
    if(sum<0)
        return 0;
    if(k==0)
        return 0;
    if(t[sum][k]!=-1)
        return t[sum][k];
    int ans=0;
    for(int i=0;i<arr.size();i++){
        if(arr[i]<=sum)
            ans=((ans%MOD)+(func(arr,sum-arr[i],k-1,t)%MOD))%MOD;
    }
    return t[sum][k]=ans%MOD;
}

int Solution::solve(int A, int B) {
    vector<vector<int>>t(1000, vector<int>(1000,-1));
    vector<int> arr = {1,2,3,4,5,6,7,8,9,0};
    int ans=0;
    for(int i=0;i<9;i++){
        ans=((ans%MOD) + (func(arr,B-arr[i],A-1,t)%MOD))%MOD;
    }
    return ans%MOD;
    
}

*****************************************************************************************
******************************************************************************************
****************************************************************************************
*****************************************************************************************
******************************************************************************************
****************************************************************************************
*****************************************************************************************
******************************************************************************************
****************************************************************************************
